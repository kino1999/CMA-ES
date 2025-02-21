#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include "Function Set.h"
#include <fstream>
using namespace std;

/**
 * @brief 经典的CMA-ES算法(按照Nikolaus Hansen的<The CMA Evolution Strategy A Tutorial>)
 */

class CMA_ES {
private:
    ///基本参数
    int dimension;
    int population_size;
    int max_iterations;
    double elite_ratio;
    double initial_step_size;
    vector<double>lower_bound;
    vector<double>upper_bound;

    ///目标函数
    function<double(const vector<double>&)> fitnessFunc;

    ///CMA-ES进化状态参数
    vector<vector<double>>population;
    vector<double>fitness;
    vector<double>current_mean;
    vector<vector<double>>covariance_matrix;
    double step_size;
    int elite_size;
    double effective_elite_size;

    ///协方差矩阵特征值和特征向量
    vector<double>eigen_values;
    vector<vector<double>>eigen_vectors;

    ///步长路径和协方差路径
    vector<double>step_path;
    vector<double>covariance_path;

    ///精英个体的权重
    vector<double>elite_weights;

    ///各因子的学习率
    double damping_factor_step_size;
    double learning_rate_step_path;
    double learning_rate_covariance_path;
    double learning_rate_rank_one;
    double learning_rate_rank_mu;
    double expectd_norm;

    mt19937 random_engine;

    ///记录最优解
    vector<double>best_solution;
    vector<double>best_fitness_record;
    double best_fitness;

    vector<double>generateIndividual() {
        vector<double>individual(dimension);
        // 1 标准正态向量 z
        vector<double> z(dimension);
        normal_distribution<double>distribution(0.0,1.0);
        for(int d = 0; d < dimension; d++){
            z[d] = distribution(random_engine);
        }

        // 1 x = m + sigma * B*D*z
        vector<double> Dz(dimension);
        for(int i = 0; i < dimension; i++) {
            Dz[i] = eigen_values[i] * z[i];
        }
        for(int r = 0; r < dimension; r++) {
            double temp = 0.0;
            for(int c = 0; c < dimension; c++) {
                temp += eigen_vectors[r][c] * Dz[c];
            }
            individual[r] = current_mean[r] + step_size * temp;
        }

        // 1 边界修正
        for(int d = 0; d < dimension; d++){
            if(individual[d] < lower_bound[d]) individual[d] = lower_bound[d];
            if(individual[d] > upper_bound[d]) individual[d] = upper_bound[d];
        }
        return individual;
    }

    int evaluation() {
        int best_index = 0;
        double local_best_fitness = numeric_limits<double>::max();
        for(int i = 0; i < population_size; i++) {
            fitness[i] = fitnessFunc(population[i]);
            if(fitness[i] < local_best_fitness) {
                local_best_fitness = fitness[i];
                best_index = i;
            }
        }
        //返回最优个体的索引
        return best_index;
    }

    vector<int> sortPopulation() {
        vector<int> sortedIdx(population_size);
        for(int i = 0; i < population_size; i++) {
            sortedIdx[i] = i;
        }
        sort(sortedIdx.begin(), sortedIdx.end(), [&](int a, int b){
            return fitness[a] < fitness[b];
        });
        return sortedIdx;
    }

    void updateEvolutionParameters(const vector<int>&sorted_index, const int current_iteration) {
        //1.更新均值向量
        vector<double>old_mean = current_mean;
        // 1.1 新均值 = \sum_i w_i * x_{i:\lambda}
        for(int d = 0; d < dimension; d++) {
            double weightedSum = 0.0;
            for(int i = 0; i < elite_size; i++) {
                weightedSum += elite_weights[i] * population[sorted_index[i]][d];
            }
            current_mean[d] = weightedSum;
        }

        // 2. 步长路径 p_sigma
        vector<double> yVector = calcInverseSqrtC(current_mean, old_mean);
        for(int d = 0; d < dimension; d++) {
            step_path[d] = (1.0 - learning_rate_step_path) * step_path[d]+ sqrt(learning_rate_step_path * (2.0 - learning_rate_step_path) * effective_elite_size)* yVector[d];
        }

        // 3. 更新步长
        double normalized_step_path = 0.0;
        for(double v : step_path) {
            normalized_step_path += (v * v);
        }
        normalized_step_path = sqrt(normalized_step_path);
        bool useHSig = (normalized_step_path / sqrt(1.0 - pow((1.0 - learning_rate_step_path), 2.0 * (current_iteration + 1))))< (1.4 + 2.0 / (dimension + 1.0));
        step_size *= exp(learning_rate_step_path / damping_factor_step_size* (normalized_step_path / expectd_norm - 1.0));

        // 4. 协方差路径 p_c
        for(int d = 0; d < dimension; d++) {
            covariance_path[d] = (1.0 - learning_rate_covariance_path) * covariance_path[d]+ (useHSig ? sqrt(learning_rate_covariance_path*(2.0 - learning_rate_covariance_path)*effective_elite_size)* yVector[d] : 0.0);
        }

        //5. rank-one 更新
        vector<vector<double>> rankOne(dimension, vector<double>(dimension, 0.0));
        for(int r = 0; r < dimension; r++) {
            for(int c = 0; c < dimension; c++) {
                rankOne[r][c] = covariance_path[r] * covariance_path[c];
            }
        }

        //6. rank-mu 更新
        vector<vector<double>> rankMu(dimension, vector<double>(dimension, 0.0));
        for(int i = 0; i < elite_size; i++) {
            vector<double> dy(dimension);
            for(int d = 0; d < dimension; d++) {
                dy[d] = (population[sorted_index[i]][d] - old_mean[d]) / step_size;
            }
            vector<double> zVec = calcInverseSqrtCComp(dy);
            for(int r = 0; r < dimension; r++) {
                for(int c = 0; c < dimension; c++) {
                    rankMu[r][c] += elite_weights[i] * zVec[r] * zVec[c];
                }
            }
        }

        //7. 更新协方差矩阵
        double sumW = 0.0;
        for(auto &w : elite_weights) sumW += w;
        for(int r = 0; r < dimension; r++) {
            for(int c = 0; c < dimension; c++) {
                covariance_matrix[r][c] =
                    (1.0 - learning_rate_rank_one - learning_rate_rank_mu * sumW) * covariance_matrix[r][c]
                    + learning_rate_rank_one * (useHSig ? 1.0 : 0.0) * rankOne[r][c]
                    + learning_rate_rank_mu * rankMu[r][c];
            }
        }


    }
    /**
     * @brief 对协方差矩阵 covarianceMatrix 做特征分解: C = B * D^2 * B^T
     */
    void eigenDecomposition() {
        vector<vector<double>> C(covariance_matrix);

        // 初始化 B = I
        for(int i = 0; i < dimension; i++) {
            for(int j = 0; j < dimension; j++) {
                eigen_vectors[i][j] = (i == j ? 1.0 : 0.0);
            }
        }

        const int maxIter = 50;
        for(int iter = 0; iter < maxIter; iter++) {
            // 找到绝对值最大的非对角元素
            int p = 0, q = 0;
            double maxVal = 0.0;
            for(int i = 0; i < dimension; i++) {
                for(int j = i + 1; j < dimension; j++) {
                    double val = fabs(C[i][j]);
                    if(val > maxVal) {
                        maxVal = val; p = i; q = j;
                    }
                }
            }
            if(maxVal < 1e-15) break;

            double diff = C[p][p] - C[q][q];
            double phi  = 0.0;
            if(fabs(diff) < 1e-30) {
                phi = (C[p][q] > 0.0 ? M_PI/4.0 : -M_PI/4.0);
            } else {
                phi = 0.5 * atan2(2.0 * C[p][q], diff);
            }
            double c = cos(phi), s = sin(phi);

            double cpp = c*c*C[p][p] + 2.0*c*s*C[p][q] + s*s*C[q][q];
            double cqq = s*s*C[p][p] - 2.0*c*s*C[p][q] + c*c*C[q][q];
            C[p][p] = cpp;
            C[q][q] = cqq;
            C[p][q] = 0.0;
            C[q][p] = 0.0;

            for(int i = 0; i < dimension; i++) {
                if(i!=p && i!=q) {
                    double Cip = c*C[i][p] + s*C[i][q];
                    double Ciq = -s*C[i][p] + c*C[i][q];
                    C[i][p] = Cip;
                    C[i][q] = Ciq;
                    C[p][i] = Cip;
                    C[q][i] = Ciq;
                }
            }
            // 更新特征向量 B
            for(int i = 0; i < dimension; i++) {
                double Bip = c*eigen_vectors[i][p] + s*eigen_vectors[i][q];
                double Biq = -s*eigen_vectors[i][p] + c*eigen_vectors[i][q];
                eigen_vectors[i][p] = Bip;
                eigen_vectors[i][q] = Biq;
            }
        }

        // 提取特征值 => eigenValues[i] = sqrt(特征值)
        for(int i = 0; i < dimension; i++) {
            double val = C[i][i];
            if(val < 0.0) val = 1e-30;
            eigen_values[i] = sqrt(val);
        }
    }
    /**
     * @brief 计算 inv_sqrtC * ((newMean - oldMean)/sigma)
     */
    vector<double> calcInverseSqrtC(const vector<double>& newMean, const vector<double>& oldMean) {
        vector<double> diff(dimension);
        for(int i = 0; i < dimension; i++) {
            diff[i] = (newMean[i] - oldMean[i]) / step_size;
        }
        return calcInverseSqrtCComp(diff);
    }
    /**
     * @brief 计算 inv_sqrtC * vec, 其中 inv_sqrtC = B * D^-1 * B^T
     */
    vector<double> calcInverseSqrtCComp(const vector<double>& vec) {
        vector<double> tmp(dimension, 0.0);
        // tmp = B^T * vec
        for(int r = 0; r < dimension; r++) {
            double sumVal = 0.0;
            for(int c = 0; c < dimension; c++) {
                sumVal += eigen_vectors[c][r] * vec[c];
            }
            tmp[r] = sumVal;
        }
        // D^-1 * tmp
        for(int i = 0; i < dimension; i++) {
            double eVal = eigen_values[i];
            if(fabs(eVal) > 1e-30) {
                tmp[i] /= eVal;
            }
        }
        // out = B * tmp
        vector<double> out(dimension, 0.0);
        for(int r = 0; r < dimension; r++) {
            double sumVal = 0.0;
            for(int c = 0; c < dimension; c++) {
                sumVal += eigen_vectors[r][c] * tmp[c];
            }
            out[r] = sumVal;
        }
        return out;
    }

public:
    CMA_ES(int dim,
        int pop_size,
        int max_iter,
        double e_radio,
        double init_step_size,
        const vector<double>& lb,
        const vector<double>&ub,
        const function<double(const vector<double>&)> &fitnessF)
    {
        //1.初始化基本参数
        dimension=dim;
        population_size=pop_size;
        max_iterations=max_iter;
        elite_ratio=e_radio;
        initial_step_size=init_step_size;
        step_size=init_step_size;
        lower_bound=lb;
        upper_bound=ub;
        fitnessFunc=fitnessF;
        population.resize(population_size);
        fitness.resize(population_size);


        //2.初始化随机数生成器
        random_engine.seed(random_device()());

        //3.检查上下界是否合法
        if (lower_bound.size()!=dimension || upper_bound.size()!=dimension) {
            cerr<<"Error: Dimension of lower_bound or upper_bound is not equal to dimension"<<endl;
            exit(1);
        }

        //4.计算精英个体的数量
        elite_size=static_cast<int>(population_size*elite_ratio);

        //5.初始化均值向量
        current_mean.resize(dimension);
        for (int i=0;i<dimension;i++) {
            uniform_real_distribution<double>distribution(lower_bound[i],upper_bound[i]);
            current_mean[i]=distribution(random_engine);
        }

        //6.初始化协方差矩阵为单位矩阵
        covariance_matrix.resize(dimension,vector<double>(dimension,0.0));
        for (int i=0;i<dimension;i++) {
            covariance_matrix[i][i]=1.0;
        }

        //7.初始化步长路径和协方差路径
        step_path.resize(dimension,0.0);
        covariance_path.resize(dimension,0.0);

        //8.初始化特征值和特征向量(使用对角矩阵表示特征向量)
        eigen_values.resize(dimension,1.0);
        eigen_vectors.resize(dimension,vector<double>(dimension,0.0));
        for (int i=0;i<dimension;i++) {
            eigen_vectors[i][i]=1.0;
        }

        //9.初始化精英个体的权重(对数式权重)
        elite_weights.resize(elite_size);
        for (int i=0;i<elite_size;i++) {
            // w_i = log(mu+0.5) - log(i+1)
            elite_weights[i]=log(elite_size + 0.5) - log(i + 1.0);
        }
        //归一化权重,使得权重之和为1
        double sumW = 0.0;
        for(auto &w : elite_weights) sumW += w;
        for(auto &w : elite_weights) w /= sumW;

        //10.计算有效精英个体数量
        effective_elite_size=0.0;
        for (double w:elite_weights) {
            effective_elite_size+=w*w;
        }
        effective_elite_size=1.0/effective_elite_size;

        //11.初始化各因子的学习率 (基于Niko Hansen的The CMA Evolution Strategy A Tutorial)
        learning_rate_step_path=(effective_elite_size + 2.0) / (dimension + effective_elite_size + 5.0); //c_sigma
        damping_factor_step_size = 1.0+ 2.0 * max(0.0, sqrt((effective_elite_size - 1.0)/(dimension + 1.0)) - 1.0)+ learning_rate_step_path;  // d_sigma
        learning_rate_covariance_path=(4.0 + effective_elite_size/dimension)/ (dimension + 4.0 + 2.0*effective_elite_size/dimension); // c_c
        learning_rate_rank_one=2.0 / (pow((dimension + 1.3), 2.0) + effective_elite_size); // c_1
        learning_rate_rank_mu= min(1.0 - learning_rate_rank_one,2.0 * (effective_elite_size - 2.0 + 1.0/effective_elite_size)/(pow(dimension + 2.0, 2.0) + effective_elite_size)); // c_mu

        //12.期望范数
        expectd_norm=sqrt(static_cast<double>(dimension))*(1.0-1.0/(4.0*dimension)+1.0/(21.0*dimension*dimension));

        //13.初始化最优解
        best_solution.resize(dimension);
        best_fitness=numeric_limits<double>::max();
        best_fitness_record.resize(max_iterations,0.0);
    }

    vector<double> run() {
        for (int i=0;i<max_iterations;i++) {
            //0.特征值分解
            eigenDecomposition();

            //1.生成种群
            for (int j=0;j<population_size;j++) {
                population[j]=generateIndividual();
            }

            //2.评估种群
            evaluation();

            //3.排序种群
            vector<int>sorted_index=sortPopulation();

            //4.更新全局最优解
            if (fitness[sorted_index[0]]<best_fitness) {
                best_fitness=fitness[sorted_index[0]];
                best_solution=population[sorted_index[0]];
            }
            best_fitness_record[i]=best_fitness;

            //5.更新进化状态参数
            updateEvolutionParameters(sorted_index,i);
        }
        //输出结果
        // cout<<"Best Fitness: "<<to_string(best_fitness)<<endl;
        return best_fitness_record;
    }
};
int main() {
    int dimension = 100;
    int population_size = 100;
    int max_iterations = 300;
    double elite_ratio = 0.25;
    double initial_step_size = 0.5;
    vector<double> lb(dimension, -5);
    vector<double> ub(dimension, 5);
    int test_amount=10;


    string path="CMA-ES_result.csv";
    ofstream ofs(path);
    if(!ofs.is_open()) {
        cerr << "Error: cannot open All_Results.csv for writing.\n";
        return 1;
    }
    // [MOD 1] 写第一行（表头）

    ofs << "generation";
    for(int g = 1; g <= max_iterations; g++){
        ofs << "," << g;
    }
    ofs << "\n";
    for (int i=0;i<test_amount;i++) {
        CMA_ES cma_es(dimension, population_size, max_iterations, elite_ratio, initial_step_size, lb, ub, RastriginFunction);
        vector<double>record=cma_es.run();
        for(int g = 0; g < max_iterations; g++){
            ofs << "," << record[g];
        }
        ofs << "\n";
    }
    ofs.close();
    return 0;
}

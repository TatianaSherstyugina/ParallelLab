#include <iostream>
#include <random>
#include <string>  
#include <stdio.h> 
#include <stdlib.h>
#include <cmath>  
#include <omp.h>  
#include <set>

using namespace std;

const double PI = acos(-1.0);
const double Eps = 0.0000000001;

struct Complex {
	double re;
	double im;
};

// Читает файл и возвращает количество бинов, попутно изменяя переданную переменную signals
int read_file(const char* filename, double*& signals) {
    FILE* f = fopen(filename, "r");

    long long N = 0;
    double tmp;
    while (fscanf(f, "%lf", &tmp) == 1) 
        N++;
    
    rewind(f);

    signals = (double*)malloc(N * sizeof(double));
    for (long long i = 0; i < N; i++) 
        fscanf(f, "%lf", &signals[i]);
    fclose(f);
    return N;
}

// Записывает сигналы в текстовом формате, который принимает gnuplot
bool write_array_for_gnuplot(const char* filename, const double* signals, long long N) {
    FILE* f = fopen(filename, "w");

    for (long long i = 0; i < N; ++i) {
        fprintf(f, "%d %.17f\n", i, signals[i]);
    }
    fclose(f);
    return true;
}

// Прямое преобразование Фурье (параллельное)
Complex* direct_fourier(const double* signals, long long N) {
    Complex* freqs = (Complex*)malloc(N * sizeof(Complex));
    #pragma omp parallel for
    for (long long k = 0; k < N; k++) {
        double re = 0.0, im = 0.0;
        for (long long i = 0; i < N; i++) {
            double angle = 2.0 * PI * k * i / N;
            re += signals[i] * cos(angle);
            im -= signals[i] * sin(angle);
        }
        freqs[k].re = re;
        freqs[k].im = im;
    }
    return freqs;
}

// Прямое преобразование Фурье (последовательное)
Complex* direct_fourier_seqly(const double* signals, long long N) {
    Complex* freqs = (Complex*)malloc(N * sizeof(Complex));
    for (long long k = 0; k < N; k++) {
        double re = 0.0, im = 0.0;
        for (long long i = 0; i < N; i++) {
            double angle = 2.0 * PI * k * i / N;
            re += signals[i] * cos(angle);
            im -= signals[i] * sin(angle);
        }
        freqs[k].re = re;
        freqs[k].im = im;
    }
    return freqs;
}

// Обратное преобразование Фурье (параллельное)
double* indirect_fourier(const Complex* freqs, long long N) {
    double* new_signals = (double*)malloc(N * sizeof(double));
    #pragma omp parallel for
    for (long long i = 0; i < N; i++) {
        double sum = 0.0;
        for (long long k = 0; k < N; k++) {
            double angle = 2.0 * PI * k * i / N;
            sum += freqs[k].re * cos(angle) - freqs[k].im * sin(angle);
        }
        new_signals[i] = sum / N;
    }
    return new_signals;
}

// Обратное преобразование Фурье (последовательное)
double* indirect_fourier_seqly(const Complex* freqs, long long N) {
    double* new_signals = (double*)malloc(N * sizeof(double));
    for (long long i = 0; i < N; i++) {
        double sum = 0.0;
        for (long long k = 0; k < N; k++) {
            double angle = 2.0 * PI * k * i / N;
            sum += freqs[k].re * cos(angle) - freqs[k].im * sin(angle);
        }
        new_signals[i] = sum / N;
    }
    return new_signals;
}

// Редукция для объединения частичных результатов в одно множество, инициализируется на каждом потоке приватным пустым множеством
#pragma omp declare reduction(set_merge : set<long long> : omp_out.merge(omp_in)) initializer(omp_priv = decltype(omp_orig)())

// Сначала находит максимальное значение параллельной редукцией, потом создает параллельной редукцией множество бинов с такой (максимальной) частотой (с погрешностью EPS) + радиусом free_radius вокруг них
set<long long> find_del_indexes(const Complex* freqs, long long N, int free_radius) {
    double max_val = 0;
    set<long long> res;

    #pragma omp parallel for reduction(max:max_val)
    for (long long k = 0; k < N; ++k) {
        const double v = hypot(freqs[k].re, freqs[k].im);
        max_val = max(max_val, v);
    }

    const double threshold = max_val - Eps;
    #pragma omp parallel for reduction(set_merge:res)
    for (long long k = 0; k < N; k++) {
        const double cur = hypot(freqs[k].re, freqs[k].im);
        if (cur > threshold) {
            long long l = max(k - (long long)free_radius, (long long)0);
            long long r = min((long long)(N - 1), k + (long long)free_radius);
            for (long long i = l; i <= r; i++) 
                    res.insert(i);
        }
    }
    return res;
}

// Сначала находит максимальное значение последовательно, потом создает последовательно множество бинов с такой (максимальной) частотой (с погрешностью EPS) + радиусом free_radius вокруг них
set<long long> find_del_indexes_seqly(const Complex* freqs, long long N, int free_radius) {
    double max_val = 0;
    set<long long> res;

    for (long long k = 0; k < N; ++k) {
        const double v = hypot(freqs[k].re, freqs[k].im);
        max_val = max(max_val, v);
    }

    const double threshold = max_val - Eps;
    for (long long k = 0; k < N; k++) {
        const double cur = hypot(freqs[k].re, freqs[k].im);
        if (cur > threshold) {
            long long l = max(k - (long long)free_radius, (long long)0);
            long long r = min((long long)(N - 1), k + (long long)free_radius);
            for (long long i = l; i <= r; i++)
                res.insert(i);
        }
    }
    return res;
}

// Параллельно затирает найденные окрестности частот
void delete_max_freqs(Complex* freqs, long long N, int free_radius) {
    const set<long long> indexes_to_delete = find_del_indexes(freqs, N, free_radius);
    #pragma omp parallel for
    for (long long idx = 0; idx < (long long)indexes_to_delete.size(); ++idx) {
        auto k = next(indexes_to_delete.begin(), idx);
        freqs[*k].re = 0.0;
        freqs[*k].im = 0.0;
    }
}

// Последовательно затирает найденные окрестности частот
void delete_max_freqs_seqly(Complex* freqs, long long N, int free_radius) {
    const set<long long> indexes_to_delete = find_del_indexes_seqly(freqs, N, free_radius);
    for (long long idx = 0; idx < (long long)indexes_to_delete.size(); ++idx) {
        auto k = next(indexes_to_delete.begin(), idx);
        freqs[*k].re = 0.0;
        freqs[*k].im = 0.0;
    }
}

// Массив амплитуд спектра по бинам
double* spectrum_amplitude(const Complex* freqs, long long N) {
    double* amps = (double*)malloc(N * sizeof(double));
    for (long long k = 0; k < N; ++k) {
        amps[k] = hypot(freqs[k].re, freqs[k].im);
    }
    return amps;
}

// Выводит максимальную амплитуду спектра, мощность и среднюю мощность
void print_spectrum_stats(const Complex* freqs, long long N, const std::string& tag) {
    double max_amp = 0.0;
    long long max_k = 0;
    double total_power = 0.0;

    for (long long k = 0; k < N; ++k) {
        double amp = hypot(freqs[k].re, freqs[k].im);
        double power = amp * amp;
        if (amp > max_amp) {
            max_amp = amp;
            max_k = k;
        }
        total_power += power;
    }

    double mean_power = total_power / N;

    std::cout << "==== Spectrum stats [" << tag << "] ====\n";
    std::cout << "Max amplitude = " << max_amp << " at bin k = " << max_k << "\n"; // 
    std::cout << "Total power   = " << total_power << "\n";
    std::cout << "Mean power    = " << mean_power << "\n";
    std::cout << "===================================\n";
}

int main() {
    double* noisy;
    int N = read_file("noisy.txt", noisy);
    double t_start, t_end;

    // Звук с шумом для gnuplot
    write_array_for_gnuplot("graph_text/noisy_gnu.txt", noisy, N);

    // Параллельное прямое преобразование Фурье
    t_start = omp_get_wtime();
    Complex* specter_par = direct_fourier(noisy, N);
    t_end = omp_get_wtime();
    std::cout << "direct_fourier (parallel) time = " << (t_end - t_start) << " s\n";

    // Частотные характеристики до удаления максимумов (для параллельного)
    print_spectrum_stats(specter_par, N, "parallel BEFORE delete");
    double* amps_par_before = spectrum_amplitude(specter_par, N);
    write_array_for_gnuplot("graph_text/spectrum_par_before.txt", amps_par_before, N);
    free(amps_par_before);

    // Параллельное удаление максимальных (с разницей в EPS) частот в некотором радиусе
    t_start = omp_get_wtime();
    delete_max_freqs(specter_par, N, 1000);
    t_end = omp_get_wtime();
    std::cout << "delete_max_freqs (parallel) time = " << (t_end - t_start) << " s\n";

    // Частотные характеристики после удаления максимумов (для параллельного)
    print_spectrum_stats(specter_par, N, "parallel AFTER delete");
    double* amps_par_after = spectrum_amplitude(specter_par, N);
    write_array_for_gnuplot("graph_text/spectrum_par_after.txt", amps_par_after, N);
    free(amps_par_after);

    // Параллельное обратное преобразование Фурье
    t_start = omp_get_wtime();
    double* res_par = indirect_fourier(specter_par, N);
    t_end = omp_get_wtime();
    std::cout << "indirect_fourier (parallel) time = " << (t_end - t_start) << " s\n\n";

    write_array_for_gnuplot("graph_text/denoised_paral_gnu.txt", res_par, N);


    // Последовательное прямое преобразование Фурье
    t_start = omp_get_wtime();
    Complex* specter_seq = direct_fourier_seqly(noisy, N);
    t_end = omp_get_wtime();
    std::cout << "direct_fourier (sequential) time = " << (t_end - t_start) << " s\n";

    // Частотные характеристики до удаления максимумов (для последовательного)
    print_spectrum_stats(specter_seq, N, "sequential BEFORE delete");
    double* amps_seq_before = spectrum_amplitude(specter_seq, N);
    write_array_for_gnuplot("graph_text/spectrum_seq_before.txt", amps_seq_before, N);
    free(amps_seq_before);

    // Последовательное удаление максимальных (с разницей в EPS) частот в некотором радиусе
    t_start = omp_get_wtime();
    delete_max_freqs_seqly(specter_seq, N, 1000);
    t_end = omp_get_wtime();
    std::cout << "delete_max_freqs (sequential) time = " << (t_end - t_start) << " s\n";

    // Частотные характеристики после удаления максимумов (для последовательного)
    print_spectrum_stats(specter_seq, N, "sequential AFTER delete");
    double* amps_seq_after = spectrum_amplitude(specter_seq, N);
    write_array_for_gnuplot("graph_text/spectrum_seq_after.txt", amps_seq_after, N);
    free(amps_seq_after);

    // Последовательное обратное преобразование Фурье
    t_start = omp_get_wtime();
    double* res_seq = indirect_fourier_seqly(specter_seq, N);
    t_end = omp_get_wtime();
    std::cout << "indirect_fourier (sequential) time = " << (t_end - t_start) << " s\n";

    write_array_for_gnuplot("graph_text/denoised_seq_gnu.txt", res_seq, N);

    return 0;
}

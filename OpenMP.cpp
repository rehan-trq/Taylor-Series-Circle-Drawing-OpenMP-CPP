// Rehan Tariq
// 22i-0965
// CS-6A


#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <SDL2/SDL.h>

// Constants for the circle
const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;
const double PI = 3.14159265358979323846;

// Number of points to draw the circle
const int NUM_POINTS = 1000;
const int TAYLOR_TERMS = 150000; // Reduced for efficiency

// Taylor series approximation for sine
inline double taylor_sin(double x, int terms) {
    double result = x;
    double term = x;
    double x_squared = x * x;
    omp_set_num_threads(4);
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < terms; i++) {
        int n = 2 * i + 1;
        term *= (-1.0 * x_squared) / ((n - 1) * n);
        result += term;
    }
    return result;
}

// Taylor series approximation for cosine
inline double taylor_cos(double x, int terms) {
    double result = 1.0;
    double term = 1.0;
    double x_squared = x * x;
     omp_set_num_threads(4);
    #pragma omp parallel for schedule(static)
    for (int i = 1; i < terms; i++) {
        int n = 2 * i;
        term *= (-1.0 * x_squared) / ((n - 1) * n);
        result += term;
    }
    return result;
}


inline double taylor_sin1(double x, int terms) {
    double result = x;
    double term = x;
    double x_squared = x * x;
    omp_set_num_threads(4);
   
    for (int i = 1; i < terms; i++) {
        int n = 2 * i + 1;
        term *= (-1.0 * x_squared) / ((n - 1) * n);
        result += term;
    }
    return result;
}

// Taylor series approximation for cosine
inline double taylor_cos1(double x, int terms) {
    double result = 1.0;
    double term = 1.0;
    double x_squared = x * x;
     omp_set_num_threads(4);
   
    for (int i = 1; i < terms; i++) {
        int n = 2 * i;
        term *= (-1.0 * x_squared) / ((n - 1) * n);
        result += term;
    }
    return result;
}


int main() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL Initialization Failed! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("Circle Drawing", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    double cx = SCREEN_WIDTH / 2.0, cy = SCREEN_HEIGHT / 2.0, r = std::min(SCREEN_WIDTH, SCREEN_HEIGHT) / 3.0;

    // Vectors for points
    std::vector<SDL_Point> serial_math(NUM_POINTS), parallel_math(NUM_POINTS);
    std::vector<SDL_Point> serial_taylor(NUM_POINTS), parallel_taylor(NUM_POINTS);

    double serial_math_time, parallel_math_time, serial_taylor_time, parallel_taylor_time;

    // SERIAL: Math Library
    double start = omp_get_wtime();
    for (int i = 0; i < NUM_POINTS; i++) {
        double t = 2.0 * PI * i / NUM_POINTS;
        serial_math[i] = { static_cast<int>(r * cos(t) + cx), static_cast<int>(r * sin(t) + cy) };
    }
    serial_math_time = omp_get_wtime() - start;

    // PARALLEL: Math Library
    start = omp_get_wtime();
    omp_set_num_threads(4);
    #pragma omp parallel for schedule(static,2)
    for (int i = 0; i < NUM_POINTS; i++) {
        double t = 2.0 * PI * i / NUM_POINTS;
        parallel_math[i] = { static_cast<int>(r * cos(t) + cx), static_cast<int>(r * sin(t) + cy) };
    }
    parallel_math_time = omp_get_wtime() - start;

    // SERIAL: Taylor Series
    start = omp_get_wtime();
    for (int i = 0; i < NUM_POINTS; i++) {
        double t = 2.0 * PI * i / NUM_POINTS;
        serial_taylor[i] = { static_cast<int>(r * taylor_cos1(t, TAYLOR_TERMS) + cx), static_cast<int>(r * taylor_sin1(t, TAYLOR_TERMS) + cy) };
    }
    serial_taylor_time = omp_get_wtime() - start;

    // PARALLEL: Taylor Series
    start = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NUM_POINTS; i++) {
        double t = 2.0 * PI * i / NUM_POINTS;
        parallel_taylor[i] = { static_cast<int>(r * taylor_cos(t, TAYLOR_TERMS) + cx), static_cast<int>(r * taylor_sin(t, TAYLOR_TERMS) + cy) };
    }
    parallel_taylor_time = omp_get_wtime() - start;

    // Draw results
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    for (const auto& p : serial_math) SDL_RenderDrawPoint(renderer, p.x, p.y);
    SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255);
    for (const auto& p : parallel_math) SDL_RenderDrawPoint(renderer, p.x, p.y);
    SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
    for (const auto& p : serial_taylor) SDL_RenderDrawPoint(renderer, p.x, p.y);
    SDL_SetRenderDrawColor(renderer, 255, 255, 0, 255);
    for (const auto& p : parallel_taylor) SDL_RenderDrawPoint(renderer, p.x, p.y);
    SDL_RenderPresent(renderer);

    // Print Results
    std::cout<< " The total taylor series terms used are :"<<TAYLOR_TERMS <<std::endl;
    std::cout << "Serial Math Library Time: " << serial_math_time * 1000 << " ms\n";
    std::cout << "Parallel Math Library Time: " << parallel_math_time * 1000 << " ms\n";
    std::cout << "Speedup (Math Library): " << serial_math_time / parallel_math_time << "x\n";
    std::cout << "Serial Taylor Series Time: " << serial_taylor_time * 1000 << " ms\n";
    std::cout << "Parallel Taylor Series Time: " << parallel_taylor_time * 1000 << " ms\n";
    std::cout << "Speedup (Taylor Series): " << serial_taylor_time / parallel_taylor_time << "x\n";
    std::cout << "Available Threads: " << omp_get_max_threads() << "\n";

    // SDL Loop
    bool quit = false;
    SDL_Event e;
    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = true;
        }
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

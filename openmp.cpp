#include <iostream>
#include <arm_neon.h>
#include <omp.h>
#include <sys/time.h>
#include <fstream>
using namespace std;
const int n = 200;
float m[n][n];
int NUM_THREADS = 3;
void generateMatrix()
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            m[i][j] = 0;
        }
        m[i][i] = 1.0;
        for (int j = i + 1; j < n; j++)
            m[i][j] = rand() % 100;
    }
    for (int i = 0; i < n; i++)
    {
        int k1 = rand() % n;
        int k2 = rand() % n;
        for (int j = 0; j < n; j++)
        {
            m[i][j] += m[0][j];
            m[k1][j] += m[k2][j];
        }
    }
}

//静态调度，无neon，一开始创建线程
void static_nonneon_stathread()
{
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++)
    {
        #pragma omp single
        {
            float tmp = m[k][k];
            for (int j = k + 1; j < n; j++)
            {
                m[k][j] = m[k][j] / tmp;
            }
            m[k][k] = 1.0;
        }
        #pragma omp for schedule(static)
        for (int i = k + 1; i < n; i++)
        {
            float tmp = m[i][k];
            for (int j = k + 1; j < n; j++)
                m[i][j] = m[i][j] - tmp * m[k][j];
            m[i][k] = 0;
        }
    }
}

//静态调度，使用neon，一开始创建线程
void static_neon_stathread()
{
    float32x4_t va = vmovq_n_f32(0);
    float32x4_t vx = vmovq_n_f32(0);
    float32x4_t vaij = vmovq_n_f32(0);
    float32x4_t vaik = vmovq_n_f32(0);
    float32x4_t vakj = vmovq_n_f32(0);
    #pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
	for (int k = 0; k < n; k++)
	{
		#pragma omp single
		{
		    float32x4_t vt=vmovq_n_f32(m[k][k]);
            int j;
			for (j = k + 1; j < n; j++)
			{
				va=vld1q_f32(&(m[k][j]) );
                va= vdivq_f32(va,vt);
                vst1q_f32(&(m[k][j]), va);
			}
			for(; j<n; j++)
            {
                m[k][j]=m[k][j]*1.0 / m[k][k];
            }
            m[k][k] = 1.0;
		}
		#pragma omp for schedule(static)
		for (int i = k + 1; i < n; i++)
		{
		    vaik=vmovq_n_f32(m[i][k]);
            int j;
			for (j = k + 1; j+4 <= n; j+=4)
			{
				vakj=vld1q_f32(&(m[k][j]));
				vaij=vld1q_f32(&(m[i][j]));
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);
				vst1q_f32(&m[i][j], vaij);
			}
			for(; j<n; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
			m[i][k] = 0;
		}
	}
}

//静态调度，使用neon，在并行任务开始前创建线程
void static_neon_dynthread()
{
    float32x4_t va = vmovq_n_f32(0);
    float32x4_t vx = vmovq_n_f32(0);
    float32x4_t vaij = vmovq_n_f32(0);
    float32x4_t vaik = vmovq_n_f32(0);
    float32x4_t vakj = vmovq_n_f32(0);
	for (int k = 0; k < n; k++)
	{
		{
		    float32x4_t vt=vmovq_n_f32(m[k][k]);
            int j;
			for (j = k + 1; j < n; j++)
			{
				va=vld1q_f32(&(m[k][j]) );
                va= vdivq_f32(va,vt);
                vst1q_f32(&(m[k][j]), va);
			}
			for(; j<n; j++)
            {
                m[k][j]=m[k][j]*1.0 / m[k][k];
            }
            m[k][k] = 1.0;
		}
        #pragma omp parallel for num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj) ,schedule(static)
		for (int i = k + 1; i < n; i++)
		{
		    vaik=vmovq_n_f32(m[i][k]);
            int j;
			for (j = k + 1; j+4 <= n; j+=4)
			{
				vakj=vld1q_f32(&(m[k][j]));
				vaij=vld1q_f32(&(m[i][j]));
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);
				vst1q_f32(&m[i][j], vaij);
			}
			for(; j<n; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
			m[i][k] = 0;
		}
	}
}

//动态调度，每次划分迭代块大小为5，使用neon，一开始创建线程
void dynamic_neon_stathread()
{
    float32x4_t va = vmovq_n_f32(0);
    float32x4_t vx = vmovq_n_f32(0);
    float32x4_t vaij = vmovq_n_f32(0);
    float32x4_t vaik = vmovq_n_f32(0);
    float32x4_t vakj = vmovq_n_f32(0);
    #pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
	for (int k = 0; k < n; k++)
	{
		#pragma omp single
		{
		    float32x4_t vt=vmovq_n_f32(m[k][k]);
            int j;
			for (j = k + 1; j < n; j++)
			{
				va=vld1q_f32(&(m[k][j]) );
                va= vdivq_f32(va,vt);
                vst1q_f32(&(m[k][j]), va);
			}
			for(; j<n; j++)
            {
                m[k][j]=m[k][j]*1.0 / m[k][k];
            }
            m[k][k] = 1.0;
		}
		#pragma omp for schedule(dynamic, 5)
		for (int i = k + 1; i < n; i++)
		{
		    vaik=vmovq_n_f32(m[i][k]);
            int j;
			for (j = k + 1; j+4 <= n; j+=4)
			{
				vakj=vld1q_f32(&(m[k][j]));
				vaij=vld1q_f32(&(m[i][j]));
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);
				vst1q_f32(&m[i][j], vaij);
			}
			for(; j<n; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
			m[i][k] = 0;
		}
	}
}

//guide调度，最小迭代块大小为1，使用neon，一开始创建线程
void guide_neon_stathread()
{
    float32x4_t va = vmovq_n_f32(0);
    float32x4_t vx = vmovq_n_f32(0);
    float32x4_t vaij = vmovq_n_f32(0);
    float32x4_t vaik = vmovq_n_f32(0);
    float32x4_t vakj = vmovq_n_f32(0);
    #pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
	for (int k = 0; k < n; k++)
	{
		#pragma omp single
		{
		    float32x4_t vt=vmovq_n_f32(m[k][k]);
            int j;
			for (j = k + 1; j < n; j++)
			{
				va=vld1q_f32(&(m[k][j]) );
                va= vdivq_f32(va,vt);
                vst1q_f32(&(m[k][j]), va);
			}
			for(; j<n; j++)
            {
                m[k][j]=m[k][j]*1.0 / m[k][k];
            }
            m[k][k] = 1.0;
		}
		#pragma omp for schedule(guided, 1)
		for (int i = k + 1; i < n; i++)
		{
		    vaik=vmovq_n_f32(m[i][k]);
            int j;
			for (j = k + 1; j+4 <= n; j+=4)
			{
				vakj=vld1q_f32(&(m[k][j]));
				vaij=vld1q_f32(&(m[i][j]));
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);
				vst1q_f32(&m[i][j], vaij);
			}
			for(; j<n; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
			m[i][k] = 0;
		}
	}
}

//两处
void doublestatic()
{
    float32x4_t va = vmovq_n_f32(0);
    float32x4_t vx = vmovq_n_f32(0);
    float32x4_t vaij = vmovq_n_f32(0);
    float32x4_t vaik = vmovq_n_f32(0);
    float32x4_t vakj = vmovq_n_f32(0);
    #pragma omp parallel num_threads(NUM_THREADS), private(va, vx, vaij, vaik,vakj)
	for (int k = 0; k < n; k++)
	{
		#pragma omp for schedule(static)
		for (int j = k + 1; j < n; j++)
		{
			m[k][j] = m[k][j] / m[k][k];
		}
		m[k][k] = 1.0;
		#pragma omp for schedule(static)
		for (int i = k + 1; i < n; i++)
		{
		    vaik=vmovq_n_f32(m[i][k]);
            int j;
			for (j = k + 1; j+4 <= n; j+=4)
			{
				vakj=vld1q_f32(&(m[k][j]));
				vaij=vld1q_f32(&(m[i][j]));
				vx=vmulq_f32(vakj,vaik);
				vaij=vsubq_f32(vaij,vx);
				vst1q_f32(&m[i][j], vaij);
			}
			for(; j<n; j++)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
			m[i][k] = 0;
		}
	}
}

void auto_simd()
{
	#pragma omp parallel num_threads(NUM_THREADS)
	for (int k = 0; k < n; k++)
	{
		#pragma omp single
		{
			float tmp = m[k][k];
			for (int j = k + 1; j < n; j++)
			{
				m[k][j] = m[k][j] / tmp;
			}
			m[k][k] = 1.0;
		}
		#pragma omp for simd
		for (int i = k + 1; i < n; i++)
		{
		    int tmp=m[i][k];
			for (int j = k + 1; j < n; j++)
				m[i][j] = m[i][j] -  tmp * m[k][j];
			m[i][k] = 0;
		}
	}
}

void add1()
{
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++)
    {
        #pragma omp single
        {
            float tmp = m[k][k];
            for (int j = k + 1; j < n; j++)
            {
                m[k][j] = m[k][j] / tmp;
            }
            m[k][k] = 1.0;
        }
        #pragma omp for schedule(dynamic,5)
        for (int i = k + 1; i < n; i++)
        {
            float tmp = m[i][k];
            for (int j = k + 1; j < n; j++)
                m[i][j] = m[i][j] - tmp * m[k][j];
            m[i][k] = 0;
        }
    }
}

void add2()
{
    #pragma omp parallel num_threads(NUM_THREADS)
    for (int k = 0; k < n; k++)
    {
        #pragma omp single
        {
            float tmp = m[k][k];
            for (int j = k + 1; j < n; j++)
            {
                m[k][j] = m[k][j] / tmp;
            }
            m[k][k] = 1.0;
        }
        #pragma omp for schedule(guide,1)
        for (int i = k + 1; i < n; i++)
        {
            float tmp = m[i][k];
            for (int j = k + 1; j < n; j++)
                m[i][j] = m[i][j] - tmp * m[k][j];
            m[i][k] = 0;
        }
    }
}

int main()
{
    ofstream out("output.txt");
    struct timeval start, over;
    double timeUse;
    for (; NUM_THREADS < 8; NUM_THREADS++)
    {
        out << n << "\t";
        out << NUM_THREADS << "\t";
        generateMatrix();
        gettimeofday(&start, NULL);
        static_nonneon_stathread();
        gettimeofday(&over, NULL);
        timeUse = (over.tv_sec - start.tv_sec) * 1000000 + (over.tv_usec - start.tv_usec);
        out << timeUse / 1000 << "\t";
    }

    out<<endl;
    for (; NUM_THREADS < 8; NUM_THREADS++)
    {
        out << n << "\t";
        out << NUM_THREADS << "\t";
        generateMatrix();
        gettimeofday(&start, NULL);
        add1();
        gettimeofday(&over, NULL);
        timeUse = (over.tv_sec - start.tv_sec) * 1000000 + (over.tv_usec - start.tv_usec);
        out << timeUse / 1000 << "\t";
    }

    out<<endl;
    for (; NUM_THREADS < 8; NUM_THREADS++)
    {
        out << n << "\t";
        out << NUM_THREADS << "\t";
        generateMatrix();
        gettimeofday(&start, NULL);
        add2();
        gettimeofday(&over, NULL);
        timeUse = (over.tv_sec - start.tv_sec) * 1000000 + (over.tv_usec - start.tv_usec);
        out << timeUse / 1000 << "\t";
    }

    out<<endl;

    // for (NUM_THREADS = 3; NUM_THREADS < 8; NUM_THREADS++)
    // {
    //     out << n << "\t";
    //     out << NUM_THREADS << "\t";
    //     generateMatrix();
    //     gettimeofday(&start, NULL);
    //     static_neon_stathread();
    //     gettimeofday(&over, NULL);
    //     timeUse = (over.tv_sec - start.tv_sec) * 1000000 + (over.tv_usec - start.tv_usec);
    //     out << timeUse / 1000 << "\t";
    // }

    // for (NUM_THREADS = 3; NUM_THREADS < 8; NUM_THREADS++)
    // {
    //     out << n << "\t";
    //     out << NUM_THREADS << "\t";
    //     generateMatrix();
    //     gettimeofday(&start, NULL);
    //     static_neon_dynthread();
    //     gettimeofday(&over, NULL);
    //     timeUse = (over.tv_sec - start.tv_sec) * 1000000 + (over.tv_usec - start.tv_usec);
    //     out << timeUse / 1000 << "\t";
    // }

    // for (NUM_THREADS = 3; NUM_THREADS < 8; NUM_THREADS++)
    // {
    //     out << n << "\t";
    //     out << NUM_THREADS << "\t";
    //     generateMatrix();
    //     gettimeofday(&start, NULL);
    //     dynamic_neon_stathread();
    //     gettimeofday(&over, NULL);
    //     timeUse = (over.tv_sec - start.tv_sec) * 1000000 + (over.tv_usec - start.tv_usec);
    //     out << timeUse / 1000 << "\t";
    // }

    // for (NUM_THREADS = 3; NUM_THREADS < 8; NUM_THREADS++)
    // {
    //     out << n << "\t";
    //     out << NUM_THREADS << "\t";
    //     generateMatrix();
    //     gettimeofday(&start, NULL);
    //     guide_neon_stathread();
    //     gettimeofday(&over, NULL);
    //     timeUse = (over.tv_sec - start.tv_sec) * 1000000 + (over.tv_usec - start.tv_usec);
    //     out << timeUse / 1000 << "\t";
    // }

    // for (NUM_THREADS = 3; NUM_THREADS < 8; NUM_THREADS++)
    // {
    //     out << n << "\t";
    //     out << NUM_THREADS << "\t";
    //     generateMatrix();
    //     gettimeofday(&start, NULL);
    //     doublestatic();
    //     gettimeofday(&over, NULL);
    //     timeUse = (over.tv_sec - start.tv_sec) * 1000000 + (over.tv_usec - start.tv_usec);
    //     out << timeUse / 1000 << "\t";
    // }

    // for (NUM_THREADS = 3; NUM_THREADS < 8; NUM_THREADS++)
    // {
    //     out << n << "\t";
    //     out << NUM_THREADS << "\t";
    //     generateMatrix();
    //     gettimeofday(&start, NULL);
    //     auto_simd();
    //     gettimeofday(&over, NULL);
    //     timeUse = (over.tv_sec - start.tv_sec) * 1000000 + (over.tv_usec - start.tv_usec);
    //     out << timeUse / 1000 << "\t";
    // }

    out.close();
}

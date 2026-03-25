# include <cmath>
# include <cstdlib>
# include <ctime>
# include <iomanip>
# include <iostream>

// 1. Swap MPI header for MDMP
# include "mdmp_pragma_interface.h"

using namespace std;

int main ( int argc, char *argv[] );
double *update ( int id, int p, int n_global, int n_local, int nsteps, 
  double dt );
void collect ( int id, int p, int n_global, int n_local, int nsteps, 
  double dt, double u_local[] );
double dudt ( double x, double t );
double exact ( double x, double t );
void timestamp ( );

//****************************************************************************80

int main ( int argc, char *argv[] )
{
  double dt = 0.0000125;
  int i_global_hi;
  int i_global_lo;
  int id;
  int n_global = 40001;
  int n_local;
  int nsteps = 400000;
  int p;
  double *u1_local;
  double wtime;

// 2. MDMP Initialization
  MDMP_COMM_INIT();
  id = MDMP_GET_RANK();
  p  = MDMP_GET_SIZE();

  if ( id == 0 ) 
  {
    timestamp ( );
    cout << "\n";
    cout << "MDMP_WAVE:\n";
    cout << "  C++ version.\n";
    cout << "  Estimate a solution of the wave equation using MDMP.\n";
    cout << "\n";
    cout << "  Using " << p << " processes.\n";
    cout << "  Using a total of " << n_global << " points.\n";
    cout << "  Using " << nsteps << " time steps of size " << dt << "\n";
    cout << "  Computing final solution at time " << dt * nsteps << "\n";
  }

  wtime = MDMP_WTIME();

  i_global_lo = (   id       * ( n_global - 1 ) ) / p;
  i_global_hi = ( ( id + 1 ) * ( n_global - 1 ) ) / p;
  if ( 0 < id )
  {
    i_global_lo = i_global_lo - 1;
  }

  n_local = i_global_hi + 1 - i_global_lo;

  u1_local = update ( id, p, n_global, n_local, nsteps, dt );

  collect ( id, p, n_global, n_local, nsteps, dt, u1_local );

// 3. MDMP Timing and Finalization
  wtime = MDMP_WTIME() - wtime;
  if ( id == 0 )
  {
    cout << "\n";
    cout << "  Elapsed wallclock time was " << wtime << " seconds\n";
  }

  delete [] u1_local;

  if ( id == 0 )
  {
    cout << "\n";
    cout << "WAVE_MDMP:\n";
    cout << "  Normal end of execution.\n";
    cout << "\n";
    timestamp ( );
  }
  
  MDMP_COMM_FINAL();

  return 0;
}

//****************************************************************************80

double *update ( int id, int p, int n_global, int n_local, int nsteps, 
  double dt ) 
{
  double alpha;
  double c;
  double dx;
  int i;
  int i_global;
  int i_global_hi;
  int i_global_lo;
  int i_local;
  int i_local_hi;
  int i_local_lo;
  int ltor = 20;
  int rtol = 10;
  double t;
  double *u0_local;
  double *u1_local;
  double *u2_local;
  double x;

  c = 1.0;
  dx = 1.0 / ( double ) ( n_global - 1 );
  alpha = c * dt / dx;

  if ( 1.0 <= fabs ( alpha ) )
  {
    if ( id == 0 )
    {
      cerr << "\n";
      cerr << "UPDATE - Warning!\n";
      cerr << "  Computation will not be stable!\n";
    }
    MDMP_COMM_FINAL();
    exit ( 1 );
  }

  i_global_lo = (   id       * ( n_global - 1 ) ) / p;
  i_global_hi = ( ( id + 1 ) * ( n_global - 1 ) ) / p;
  if ( 0 < id )
  {
    i_global_lo = i_global_lo - 1;
  }

  i_local_lo = 0;
  i_local_hi = i_global_hi - i_global_lo;

  u0_local = new double[n_local];
  u1_local = new double[n_local];
  u2_local = new double[n_local];

  t = 0.0;
  for ( i_global = i_global_lo; i_global <= i_global_hi; i_global++ ) 
  {
    x = ( double ) ( i_global ) / ( double ) ( n_global - 1 );
    i_local = i_global - i_global_lo;
    u1_local[i_local] = exact ( x, t );
  }

  for ( i_local = i_local_lo; i_local <= i_local_hi; i_local++ )
  {
    u0_local[i_local] = u1_local[i_local];
  }

  for ( i = 1; i <= nsteps; i++ )
  {
    t = dt * ( double ) i;

    if ( i == 1 )
    {
      for ( i_local = i_local_lo + 1; i_local < i_local_hi; i_local++ ) 
      {
        i_global = i_global_lo + i_local;
        x = ( double ) ( i_global ) / ( double ) ( n_global - 1 );
        u2_local[i_local] = 
          +         0.5 * alpha * alpha   * u1_local[i_local-1]
          + ( 1.0 -       alpha * alpha ) * u1_local[i_local] 
          +         0.5 * alpha * alpha   * u1_local[i_local+1]
          +                            dt * dudt ( x, t );
      }
    }
    else
    {
      for ( i_local = i_local_lo + 1; i_local < i_local_hi; i_local++ ) 
      {
        u2_local[i_local] = 
          +               alpha * alpha   * u1_local[i_local-1]
          + 2.0 * ( 1.0 - alpha * alpha ) * u1_local[i_local] 
          +               alpha * alpha   * u1_local[i_local+1]
          -                                 u0_local[i_local];
      }
    }

// 4. Converted Boundary Exchange to MDMP Abstractions
    int left_neighbor  = (id > 0) ? id - 1 : MDMP_IGNORE;
    int right_neighbor = (id < p - 1) ? id + 1 : MDMP_IGNORE;

    MDMP_COMMREGION_BEGIN();
    
    // Left Exchange
    MDMP_SEND ( &u2_local[i_local_lo+1], 1, id, left_neighbor, rtol );
    MDMP_RECV ( &u2_local[i_local_lo],   1, id, left_neighbor, ltor );

    // Right Exchange
    MDMP_SEND ( &u2_local[i_local_hi-1], 1, id, right_neighbor, ltor );
    MDMP_RECV ( &u2_local[i_local_hi],   1, id, right_neighbor, rtol );

    // Because MDMP isolates the network delay, we can compute the exact 
    // physics boundaries LOCALLY while the messages are in flight!
    if ( id == 0 ) u2_local[i_local_lo] = exact ( 0.0, t );
    if ( id == p - 1 ) u2_local[i_local_hi] = exact ( 1.0, t );

    MDMP_COMMREGION_END();

    for ( i_local = i_local_lo; i_local <= i_local_hi; i_local++ )
    {
      u0_local[i_local] = u1_local[i_local];
      u1_local[i_local] = u2_local[i_local];
    }
  }

  delete [] u0_local;
  delete [] u2_local;

  return u1_local;
}

//****************************************************************************80

void collect ( int id, int p, int n_global, int n_local, int nsteps, 
  double dt, double u_local[] ) 
{
  int buffer[2];
  int collect1 = 10;
  int collect2 = 20;
  int i;
  int i_global;
  int i_global_hi;
  int i_global_lo;
  int i_local;
  int i_local_hi;
  int i_local_lo;
  int n_local2;
  double t;
  double *u_global;
  double x;

  i_global_lo = (   id       * ( n_global - 1 ) ) / p;
  i_global_hi = ( ( id + 1 ) * ( n_global - 1 ) ) / p;
  if ( 0 < id ) i_global_lo = i_global_lo - 1;

  i_local_lo = 0;
  i_local_hi = i_global_hi - i_global_lo;

  if ( id == 0 )
  {
    u_global = new double[n_global];
    for ( i_local = i_local_lo; i_local <= i_local_hi; i_local++ )
    {
      i_global = i_global_lo + i_local - i_local_lo;
      u_global[i_global] = u_local[i_local];
    }

    for ( i = 1; i < p; i++ ) 
    {
// 5. Converted Manual Point-to-Point Collection
      MDMP_COMMREGION_BEGIN();
      MDMP_RECV ( buffer, 2, 0, i, collect1 );
      MDMP_COMMREGION_END();

      i_global_lo = buffer[0];
      n_local2 = buffer[1];

      MDMP_COMMREGION_BEGIN();
      MDMP_RECV ( &u_global[i_global_lo], n_local2, 0, i, collect2 );
      MDMP_COMMREGION_END();
    }

    t = dt * ( double ) nsteps;
    cout << "\n";
    cout << "    I      X     F(X)   Exact\n";
    cout << "\n";
    for ( i_global = 0; i_global < n_global; i_global++) 
    {
      x = ( double ) ( i_global ) / ( double ) ( n_global - 1 );
      // Only print out the last element so that printing does not slow down any timings
      if ( i_global == n_global - 1 ) {
        cout << "  " << setw(3) << i_global
             << "  " << setprecision(3) << setw(6) << x
             << "  " << setprecision(3) << setw(6) << u_global[i_global]
             << "  " << setprecision(3) << setw(6) << exact ( x, t ) << "\n";
        }
    }

    delete [] u_global;
  }
  else
  {
    buffer[0] = i_global_lo;
    buffer[1] = n_local;
    
    // Workers push both messages into a single region!
    MDMP_COMMREGION_BEGIN();
    MDMP_SEND ( buffer, 2, id, 0, collect1 );
    MDMP_SEND ( u_local, n_local, id, 0, collect2 );
    MDMP_COMMREGION_END();
  }
}

// (The remaining math functions remain exactly the same as the original)
double exact ( double x, double t ) {
  const double c = 1.0;
  const double pi = 3.141592653589793;
  return sin ( 2.0 * pi * ( x - c * t ) );
}

double dudt ( double x, double t ) {
  const double c = 1.0;
  const double pi = 3.141592653589793;
  return - 2.0 * pi * c * cos ( 2.0 * pi * ( x - c * t ) );
}

void timestamp ( ) {
# define TIME_SIZE 40
  static char time_buffer[TIME_SIZE];
  const struct std::tm *tm_ptr;
  std::time_t now;
  now = std::time ( NULL );
  tm_ptr = std::localtime ( &now );
  std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );
  std::cout << time_buffer << "\n";
  return;
# undef TIME_SIZE
}

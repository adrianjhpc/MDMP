# include <cmath>
# include <cstdlib>
# include <ctime>
# include <iomanip>
# include <iostream>

// Include the MDMP interface
# include "mdmp_interface.h"

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

  // MDMP Initialization
  MDMP_COMM_INIT();
  id = MDMP_GET_RANK();
  p  = MDMP_GET_SIZE();

  if ( id == 0 ) 
  {
    timestamp ( );
    cout << "\n";
    cout << "WAVE_MDMP (Declarative RegCom Version):\n";
    cout << "  Estimate a solution of the wave equation using MDMP.\n";
    cout << "  Using " << p << " processes.\n";
    cout << "  Using a total of " << n_global << " points.\n";
    cout << "  Using " << nsteps << " time steps.\n";
    cout << "  Computing final solution at time " << dt * nsteps << "\n";
  }

  wtime = MDMP_WTIME();

  i_global_lo = (   id       * ( n_global - 1 ) ) / p;
  i_global_hi = ( ( id + 1 ) * ( n_global - 1 ) ) / p;
  if ( 0 < id ) i_global_lo = i_global_lo - 1;

  n_local = i_global_hi + 1 - i_global_lo;

  u1_local = update ( id, p, n_global, n_local, nsteps, dt );

  collect ( id, p, n_global, n_local, nsteps, dt, u1_local );

  wtime = MDMP_WTIME() - wtime;
  double max_time = 0.0;
  
  MDMP_REDUCE(&wtime, &max_time, 1, 0, MDMP_MAX);
  
  if ( id == 0 )
  {
    cout << "\n  Elapsed wallclock time was " << max_time << " seconds\n";
  }

  delete [] u1_local;

  if ( id == 0 )
  {
    cout << "\nWAVE_MDMP: Normal end of execution.\n";
    timestamp ( );
  }
  
  MDMP_COMM_FINAL();
  return 0;
}

//****************************************************************************80

double *update ( int id, int p, int n_global, int n_local, int nsteps, 
  double dt ) 
{
  double alpha, c, dx, t, x;
  int i, i_global, i_global_hi, i_global_lo, i_local, i_local_hi, i_local_lo;
  int ltor = 20, rtol = 10;
  double *u0_local, *u1_local, *u2_local;

  c = 1.0;
  dx = 1.0 / ( double ) ( n_global - 1 );
  alpha = c * dt / dx;

  if ( 1.0 <= fabs ( alpha ) )
  {
    if ( id == 0 ) cerr << "UPDATE - Stability Warning!\n";
    MDMP_COMM_FINAL();
    exit ( 1 );
  }

  i_global_lo = (id * (n_global - 1)) / p;
  i_global_hi = ((id + 1) * (n_global - 1)) / p;
  if ( 0 < id ) i_global_lo = i_global_lo - 1;

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
    u0_local[i_local] = u1_local[i_local];
  }

  for ( i = 1; i <= nsteps; i++ )
  {
    t = dt * ( double ) i;

    // Interior points calculation
    if ( i == 1 )
    {
      for ( i_local = i_local_lo + 1; i_local < i_local_hi; i_local++ ) 
      {
        i_global = i_global_lo + i_local;
        x = ( double ) ( i_global ) / ( double ) ( n_global - 1 );
        u2_local[i_local] = 0.5 * alpha * alpha * u1_local[i_local-1]
          + ( 1.0 - alpha * alpha ) * u1_local[i_local] 
          + 0.5 * alpha * alpha * u1_local[i_local+1] + dt * dudt ( x, t );
      }
    }
    else
    {
      for ( i_local = i_local_lo + 1; i_local < i_local_hi; i_local++ ) 
      {
        u2_local[i_local] = alpha * alpha * u1_local[i_local-1]
          + 2.0 * ( 1.0 - alpha * alpha ) * u1_local[i_local] 
          + alpha * alpha * u1_local[i_local+1] - u0_local[i_local];
      }
    }

    int left_neighbour  = (id > 0) ? id - 1 : MDMP_IGNORE;
    int right_neighbour = (id < p - 1) ? id + 1 : MDMP_IGNORE;

    MDMP_COMMREGION_BEGIN();
    
    // Register the four boundary operations (ghost cell exchange)
    MDMP_REGISTER_SEND ( &u2_local[i_local_lo+1], 1, id, left_neighbour, rtol );
    MDMP_REGISTER_RECV ( &u2_local[i_local_lo],   1, id, left_neighbour, ltor );
    MDMP_REGISTER_SEND ( &u2_local[i_local_hi-1], 1, id, right_neighbour, ltor );
    MDMP_REGISTER_RECV ( &u2_local[i_local_hi],   1, id, right_neighbour, rtol );

    // Commit the batch
    MDMP_COMMIT();

    // Local physics boundary conditions (CPU works while NIC communicates)
    if ( id == 0 ) u2_local[i_local_lo] = exact ( 0.0, t );
    if ( id == p - 1 ) u2_local[i_local_hi] = exact ( 1.0, t );

    MDMP_COMMREGION_END();

    // Update time-level buffers
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
  int collect1 = 10, collect2 = 20;
  int i, i_global, i_global_hi, i_global_lo, i_local, i_local_hi, i_local_lo, n_local2;
  double t, *u_global, x;

  i_global_lo = (id * (n_global - 1)) / p;
  i_global_hi = ((id + 1) * (n_global - 1)) / p;
  if ( 0 < id ) i_global_lo = i_global_lo - 1;

  i_local_lo = 0;
  i_local_hi = i_global_hi - i_global_lo;

  if ( id == 0 )
  {
    u_global = new double[n_global];
    for ( i_local = i_local_lo; i_local <= i_local_hi; i_local++ )
    {
      u_global[i_global_lo + i_local] = u_local[i_local];
    }

    for ( i = 1; i < p; i++ ) 
    {
      // Declarative Receive of metadata
      MDMP_COMMREGION_BEGIN();
      MDMP_REGISTER_RECV ( buffer, 2, 0, i, collect1 );
      MDMP_COMMIT();
      MDMP_COMMREGION_END();

      i_global_lo = buffer[0];
      n_local2 = buffer[1];

      // Declarative Receive of data segment
      MDMP_COMMREGION_BEGIN();
      MDMP_REGISTER_RECV ( &u_global[i_global_lo], n_local2, 0, i, collect2 );
      MDMP_COMMIT();
      MDMP_COMMREGION_END();
    }

    // Print out final result for Rank 0
    t = dt * (double)nsteps;
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
    
    // Worker registers both sends and commits them together
    MDMP_COMMREGION_BEGIN();
    MDMP_REGISTER_SEND ( buffer, 2, id, 0, collect1 );
    MDMP_REGISTER_SEND ( u_local, n_local, id, 0, collect2 );
    MDMP_COMMIT();
    MDMP_COMMREGION_END();
  }
}

// Math functions
double exact ( double x, double t ) {
  const double c = 1.0, pi = 3.141592653589793;
  return sin ( 2.0 * pi * ( x - c * t ) );
}

double dudt ( double x, double t ) {
  const double c = 1.0, pi = 3.141592653589793;
  return - 2.0 * pi * c * cos ( 2.0 * pi * ( x - c * t ) );
}

void timestamp ( ) {
# define TIME_SIZE 40
  static char time_buffer[TIME_SIZE];
  const struct std::tm *tm_ptr;
  std::time_t now = std::time ( NULL );
  tm_ptr = std::localtime ( &now );
  std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );
  std::cout << time_buffer << "\n";
}

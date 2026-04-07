# include <cstdlib>
# include <ctime>
# include <iomanip>
# include <iostream>

// Replaced mpi.h with the pure MDMP interface
# include "mdmp_interface.h"

using namespace std;

int main ( int argc, char *argv[] );
int circuit_value ( int n, int bvec[] );
void i4_to_bvec ( int i4, int n, int bvec[] );
void timestamp ( );

//****************************************************************************80

int main ( int argc, char *argv[] )
{
# define N 23

  int bvec[N];
  int i;
  int id;
  int ihi;
  int ilo;
  int j;
  int n = N;
  int p;
  int solution_num;
  int solution_num_local;
  int value;
  double wtime;

  // Initialize MDMP.
  MDMP_COMM_INIT();
  id = MDMP_GET_RANK();
  p = MDMP_GET_SIZE();

  if ( id == 0 ) 
  {
    cout << "\n";
    timestamp ( );
    cout << "\n";
    cout << "SATISFY_MDMP\n";
    cout << "  C++/MDMP Declarative Inspector-Executor version\n";
    cout << "  exhaustive search of all 2^N possibilities.\n";
  }

  ilo = 0;
  ihi = 1;
  for ( i = 1; i <= n; i++ )
  {
    ihi = ihi * 2;
  }

  if ( id == 0 )
  {
    cout << "\n";
    cout << "  The number of logical variables is N = " << n << "\n";
    cout << "  The number of input vectors to check is " << ihi << "\n";
    cout << "\n";
    cout << "   # Processor       Index    ---------Input Values------------------------\n";
  }

  // Domain Decomposition
  int ilo2 = ( ( p - id     ) * ilo + ( id     ) * ihi ) / p;
  int ihi2 = ( ( p - id - 1 ) * ilo + ( id + 1 ) * ihi ) / p;

  cout << "  Processor " << id << " iterates from " << ilo2 << " <= I < " << ihi2 << "\n";

  solution_num_local = 0;

  wtime = MDMP_WTIME();

  // Exhaustive search loop
  for ( i = ilo2; i < ihi2; i++ )
  {
    i4_to_bvec ( i, n, bvec );
    value = circuit_value ( n, bvec );

    if ( value == 1 )
    {
      solution_num_local = solution_num_local + 1;
      // Note: Serialized output in MPI/MDMP can be messy; 
      // typically we'd buffer this, but we'll keep the original logic.
      cout << "  " << setw(2) << solution_num_local
           << "  " << setw(8) << id
           << "  " << setw(10) << i;

      for ( j = 0; j < n; j++ )
      {
        cout << " " << bvec[j];
      }
      cout << "\n";
    }
  }

  MDMP_COMMREGION_BEGIN();

  // Register the intent to sum all local counts into solution_num at Rank 0
  MDMP_REGISTER_REDUCE(&solution_num_local, &solution_num, 1, 0, MDMP_SUM);

  // Commit the batch
  MDMP_COMMIT();

  MDMP_COMMREGION_END();

  wtime = MDMP_WTIME() - wtime;
  double max_time = 0.0;
  MDMP_REDUCE(&wtime, &max_time, 1, 0, MDMP_MAX);
  
  if ( id == 0 )
  {
    cout << "\n";
    cout << "  Number of solutions found was " << solution_num << "\n";
    cout << "  Elapsed wall clock time (seconds) " << max_time << "\n";
  }

  MDMP_COMM_FINAL();

  if ( id == 0 )
  {
    cout << "\n";
    cout << "SATISFY_MDMP: Normal end of execution.\n";
    timestamp ( );
  }

  return 0;
# undef N
}

//****************************************************************************80

int circuit_value ( int n, int bvec[] )

//****************************************************************************80
//
//  Purpose:
//
//    CIRCUIT_VALUE returns the value of a circuit for a given input set.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    20 March 2009
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Michael Quinn,
//    Parallel Programming in C with MPI and OpenMP,
//    McGraw-Hill, 2004,
//    ISBN13: 978-0071232654,
//    LC: QA76.73.C15.Q55.
//
//  Parameters:
//
//    Input, int N, the length of the input vector.
//
//    Input, int BVEC[N], the binary inputs.
//
//    Output, int CIRCUIT_VALUE, the output of the circuit.
//
{
  int value;

  value = 
       (  bvec[0]  ||  bvec[1]  )
    && ( !bvec[1]  || !bvec[3]  )
    && (  bvec[2]  ||  bvec[3]  )
    && ( !bvec[3]  || !bvec[4]  )
    && (  bvec[4]  || !bvec[5]  )
    && (  bvec[5]  || !bvec[6]  )
    && (  bvec[5]  ||  bvec[6]  )
    && (  bvec[6]  || !bvec[15] )
    && (  bvec[7]  || !bvec[8]  )
    && ( !bvec[7]  || !bvec[13] )
    && (  bvec[8]  ||  bvec[9]  )
    && (  bvec[8]  || !bvec[9]  )
    && ( !bvec[9]  || !bvec[10] )
    && (  bvec[9]  ||  bvec[11] )
    && (  bvec[10] ||  bvec[11] )
    && (  bvec[12] ||  bvec[13] )
    && (  bvec[13] || !bvec[14] )
    && (  bvec[14] ||  bvec[15] )
    && (  bvec[14] ||  bvec[16] )
    && (  bvec[17] ||  bvec[1]  )
    && (  bvec[18] || !bvec[0]  )
    && (  bvec[19] ||  bvec[1]  )
    && (  bvec[19] || !bvec[18] )
    && ( !bvec[19] || !bvec[9]  )
    && (  bvec[0]  ||  bvec[17] )
    && ( !bvec[1]  ||  bvec[20] )
    && ( !bvec[21] ||  bvec[20] )
    && ( !bvec[22] ||  bvec[20] )
    && ( !bvec[21] || !bvec[20] )
    && (  bvec[22] || !bvec[20] );

  return value;
}
//****************************************************************************80

void i4_to_bvec ( int i4, int n, int bvec[] )

//****************************************************************************80
//
//  Purpose:
//
//    I4_TO_BVEC converts an integer into a binary vector.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    20 March 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int I4, the integer.
//
//    Input, int N, the dimension of the vector.
//
//    Output, int BVEC[N], the vector of binary remainders.
//
{
  int i;

  for ( i = n - 1; 0 <= i; i-- )
  {
    bvec[i] = i4 % 2;
    i4 = i4 / 2;
  }

  return;
}
//****************************************************************************80

void timestamp ( )

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
//  Example:
//
//    31 May 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    24 September 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
//
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
}

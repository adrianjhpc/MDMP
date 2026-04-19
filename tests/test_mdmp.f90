program test_mdmp_f
  use iso_c_binding, only: c_double
  use mdmp_interface
  implicit none

  integer :: mdmp_rank, mdmp_size
  integer :: ierr  
  integer, parameter :: size = 10000
  real(c_double), dimension(size) :: data, result
  integer :: i

  call mdmp_comm_init()

  mdmp_rank = mdmp_get_rank()
  mdmp_size = mdmp_get_size()

  if (mdmp_rank == 0) then
     print *, "=== Simple MDMP Fortran Test ==="
  end if
  print *, "Rank ", mdmp_rank, " of ", mdmp_size

  ! Initialize data
  do i = 1, size
     data(i) = real(i - 1, c_double)
     result(i) = 0.0_c_double
  end do

  ! Ping-pong communication 
  ! Functions must be assigned in Fortran
  ierr = mdmp_send(data, size, 0, 1, 0)   ! Rank 0 sends to Rank 1
  ierr = mdmp_recv(result, size, 1, 0, 0) ! Rank 1 recvs from Rank 0
  ierr = mdmp_send(result, size, 1, 0, 0) ! Rank 1 sends to Rank 0
  ierr = mdmp_recv(result, size, 0, 1, 0) ! Rank 0 recvs from Rank 1

  ! Math loop
  do i = 1, size
     result(i) = result(i) * result(i) * result(i)
  end do

  call mdmp_comm_sync()
  call mdmp_comm_final()

  ! Print the result from Rank 0 
  if (mdmp_rank == 0) then
     print *, "First result: ", result(11)
     print *, "Simple MDMP Fortran test completed successfully!"
  end if

end program test_mdmp_f

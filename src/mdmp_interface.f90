module mdmp_interface
  use iso_c_binding
  implicit none
  private

  ! ------------------------------------------------------------------
  ! Public Constants
  ! ------------------------------------------------------------------
  integer(c_int), parameter, public :: MDMP_IGNORE = -2
  integer(c_int), parameter, public :: MDMP_SUM    = 0
  integer(c_int), parameter, public :: MDMP_MAX    = 1
  integer(c_int), parameter, public :: MDMP_MIN    = 2
  integer(c_int), parameter, public :: MDMP_PROD   = 3

  ! Internal Type Codes matching the C/C++ MDMPTypeTraits
  integer(c_int), parameter :: MDMP_TYPE_INT    = 0
  integer(c_int), parameter :: MDMP_TYPE_DOUBLE = 1
  integer(c_int), parameter :: MDMP_TYPE_FLOAT  = 2
  integer(c_int), parameter :: MDMP_TYPE_CHAR   = 3
  integer(c_int), parameter :: MDMP_TYPE_INT64  = 5

  ! ------------------------------------------------------------------
  ! Public API Export
  ! ------------------------------------------------------------------
  public :: mdmp_comm_init, mdmp_comm_final
  public :: mdmp_commregion_begin, mdmp_commregion_end
  public :: mdmp_comm_sync
  public :: mdmp_get_size, mdmp_get_rank
  public :: mdmp_wtime, mdmp_abort, mdmp_set_debug
  public :: mdmp_commit

  public :: mdmp_send, mdmp_recv, mdmp_reduce, mdmp_gather
  public :: mdmp_allreduce, mdmp_allgather, mdmp_bcast
  public :: mdmp_register_send, mdmp_register_recv
  public :: mdmp_register_reduce, mdmp_register_gather
  public :: mdmp_register_allreduce, mdmp_register_allgather
  public :: mdmp_register_bcast

  ! ------------------------------------------------------------------
  ! Direct C-Bindings to the backend runtime (__mdmp_marker_*)
  ! ------------------------------------------------------------------
  ! ------------------------------------------------------------------
  ! Direct C-Bindings to the backend runtime (__mdmp_marker_*)
  ! ------------------------------------------------------------------
  interface
    subroutine c_mdmp_init() bind(C, name="__mdmp_marker_init")
    end subroutine
    
    subroutine c_mdmp_final() bind(C, name="__mdmp_marker_final")
    end subroutine
    
    function c_mdmp_get_size() bind(C, name="__mdmp_marker_get_size")
      import :: c_int
      integer(c_int) :: c_mdmp_get_size
    end function
    
    function c_mdmp_get_rank() bind(C, name="__mdmp_marker_get_rank")
      import :: c_int
      integer(c_int) :: c_mdmp_get_rank
    end function
    
    subroutine c_mdmp_commregion_begin() bind(C, name="__mdmp_marker_commregion_begin")
    end subroutine
    
    subroutine c_mdmp_commregion_end() bind(C, name="__mdmp_marker_commregion_end")
    end subroutine
    
    subroutine c_mdmp_sync() bind(C, name="__mdmp_marker_sync")
    end subroutine
    
    function c_mdmp_wtime() bind(C, name="__mdmp_marker_wtime")
      import :: c_double
      real(c_double) :: c_mdmp_wtime
    end function
    
    subroutine c_mdmp_set_debug(enable) bind(C, name="__mdmp_marker_set_debug")
      import :: c_int
      integer(c_int), value :: enable
    end subroutine
    
    subroutine c_mdmp_abort(err_code) bind(C, name="__mdmp_marker_abort")
      import :: c_int
      integer(c_int), value :: err_code
    end subroutine
    
    function c_mdmp_commit() bind(C, name="__mdmp_marker_commit")
      import :: c_int
      integer(c_int) :: c_mdmp_commit
    end function

    ! --- Imperative Ops ---
    function c_mdmp_send(buf, count, dtype, byte_size, sender, dest, tag) bind(C, name="__mdmp_marker_send")
      import :: c_int, c_size_t
      type(*), dimension(*) :: buf
      integer(c_size_t), value :: count, byte_size
      integer(c_int), value :: dtype, sender, dest, tag
      integer(c_int) :: c_mdmp_send
    end function
    
    function c_mdmp_recv(buf, count, dtype, byte_size, receiver, src, tag) bind(C, name="__mdmp_marker_recv")
      import :: c_int, c_size_t
      type(*), dimension(*) :: buf
      integer(c_size_t), value :: count, byte_size
      integer(c_int), value :: dtype, receiver, src, tag
      integer(c_int) :: c_mdmp_recv
    end function
    
    function c_mdmp_reduce(in_buf, out_buf, count, dtype, byte_size, root, op) bind(C, name="__mdmp_marker_reduce")
      import :: c_int, c_size_t
      type(*), dimension(*) :: in_buf, out_buf
      integer(c_size_t), value :: count, byte_size
      integer(c_int), value :: dtype, root, op
      integer(c_int) :: c_mdmp_reduce
    end function
    
    function c_mdmp_gather(send_buf, send_count, recv_buf, dtype, byte_size, root) bind(C, name="__mdmp_marker_gather")
      import :: c_int, c_size_t
      type(*), dimension(*) :: send_buf, recv_buf
      integer(c_size_t), value :: send_count, byte_size
      integer(c_int), value :: dtype, root
      integer(c_int) :: c_mdmp_gather
    end function
    
    function c_mdmp_allreduce(in_buf, out_buf, count, dtype, byte_size, op) bind(C, name="__mdmp_marker_allreduce")
      import :: c_int, c_size_t
      type(*), dimension(*) :: in_buf, out_buf
      integer(c_size_t), value :: count, byte_size
      integer(c_int), value :: dtype, op
      integer(c_int) :: c_mdmp_allreduce
    end function
    
    function c_mdmp_allgather(in_buf, count, out_buf, dtype, byte_size) bind(C, name="__mdmp_marker_allgather")
      import :: c_int, c_size_t
      type(*), dimension(*) :: in_buf, out_buf
      integer(c_size_t), value :: count, byte_size
      integer(c_int), value :: dtype
      integer(c_int) :: c_mdmp_allgather
    end function
    
    function c_mdmp_bcast(buf, count, dtype, byte_size, root) bind(C, name="__mdmp_marker_bcast")
      import :: c_int, c_size_t
      type(*), dimension(*) :: buf
      integer(c_size_t), value :: count, byte_size
      integer(c_int), value :: dtype, root
      integer(c_int) :: c_mdmp_bcast
    end function

    ! --- Declarative Ops ---
    subroutine c_mdmp_register_send(buf, count, dtype, byte_size, sender, dest, tag) bind(C, name="__mdmp_marker_register_send")
      import :: c_int, c_size_t
      type(*), dimension(*) :: buf
      integer(c_size_t), value :: count, byte_size
      integer(c_int), value :: dtype, sender, dest, tag
    end subroutine
    
    subroutine c_mdmp_register_recv(buf, count, dtype, byte_size, receiver, src, tag) bind(C, name="__mdmp_marker_register_recv")
      import :: c_int, c_size_t
      type(*), dimension(*) :: buf
      integer(c_size_t), value :: count, byte_size
      integer(c_int), value :: dtype, receiver, src, tag
    end subroutine
    
    subroutine c_mdmp_register_reduce(in_buf, out_buf, count, dtype, byte_size, root, op) bind(C, name="__mdmp_marker_register_reduce")
      import :: c_int, c_size_t
      type(*), dimension(*) :: in_buf, out_buf
      integer(c_size_t), value :: count, byte_size
      integer(c_int), value :: dtype, root, op
    end subroutine
    
    subroutine c_mdmp_register_gather(send_buf, send_count, recv_buf, dtype, byte_size, root) bind(C, name="__mdmp_marker_register_gather")
      import :: c_int, c_size_t
      type(*), dimension(*) :: send_buf, recv_buf
      integer(c_size_t), value :: send_count, byte_size
      integer(c_int), value :: dtype, root
    end subroutine
    
    subroutine c_mdmp_register_allreduce(in_buf, out_buf, count, dtype, byte_size, op) bind(C, name="__mdmp_marker_register_allreduce")
      import :: c_int, c_size_t
      type(*), dimension(*) :: in_buf, out_buf
      integer(c_size_t), value :: count, byte_size
      integer(c_int), value :: dtype, op
    end subroutine
    
    subroutine c_mdmp_register_allgather(in_buf, count, out_buf, dtype, byte_size) bind(C, name="__mdmp_marker_register_allgather")
      import :: c_int, c_size_t
      type(*), dimension(*) :: in_buf, out_buf
      integer(c_size_t), value :: count, byte_size
      integer(c_int), value :: dtype
    end subroutine
    
    subroutine c_mdmp_register_bcast(buf, count, dtype, byte_size, root) bind(C, name="__mdmp_marker_register_bcast")
      import :: c_int, c_size_t
      type(*), dimension(*) :: buf
      integer(c_size_t), value :: count, byte_size
      integer(c_int), value :: dtype, root
    end subroutine
  end interface  

  ! ------------------------------------------------------------------
  ! Generic Interfaces (Replaces C Macros for Type Deduction)
  ! ------------------------------------------------------------------
  interface mdmp_send
    module procedure mdmp_send_r64, mdmp_send_i32, mdmp_send_r32, mdmp_send_i64
  end interface
  interface mdmp_recv
    module procedure mdmp_recv_r64, mdmp_recv_i32, mdmp_recv_r32, mdmp_recv_i64
  end interface
  interface mdmp_reduce
    module procedure mdmp_reduce_r64, mdmp_reduce_i32, mdmp_reduce_r32, mdmp_reduce_i64
  end interface
  interface mdmp_gather
    module procedure mdmp_gather_r64, mdmp_gather_i32, mdmp_gather_r32, mdmp_gather_i64
  end interface
  interface mdmp_allreduce
    module procedure mdmp_allreduce_r64, mdmp_allreduce_i32, mdmp_allreduce_r32, mdmp_allreduce_i64
  end interface
  interface mdmp_allgather
    module procedure mdmp_allgather_r64, mdmp_allgather_i32, mdmp_allgather_r32, mdmp_allgather_i64
  end interface
  interface mdmp_bcast
    module procedure mdmp_bcast_r64, mdmp_bcast_i32, mdmp_bcast_r32, mdmp_bcast_i64
  end interface
  
  interface mdmp_register_send
    module procedure mdmp_register_send_r64, mdmp_register_send_i32, mdmp_register_send_r32, mdmp_register_send_i64
  end interface
  interface mdmp_register_recv
    module procedure mdmp_register_recv_r64, mdmp_register_recv_i32, mdmp_register_recv_r32, mdmp_register_recv_i64
  end interface
  interface mdmp_register_reduce
    module procedure mdmp_register_reduce_r64, mdmp_register_reduce_i32, mdmp_register_reduce_r32, mdmp_register_reduce_i64
  end interface
  interface mdmp_register_gather
    module procedure mdmp_register_gather_r64, mdmp_register_gather_i32, mdmp_register_gather_r32, mdmp_register_gather_i64
  end interface
  interface mdmp_register_allreduce
    module procedure mdmp_register_allreduce_r64, mdmp_register_allreduce_i32, mdmp_register_allreduce_r32, mdmp_register_allreduce_i64
  end interface
  interface mdmp_register_allgather
    module procedure mdmp_register_allgather_r64, mdmp_register_allgather_i32, mdmp_register_allgather_r32, mdmp_register_allgather_i64
  end interface
  interface mdmp_register_bcast
    module procedure mdmp_register_bcast_r64, mdmp_register_bcast_i32, mdmp_register_bcast_r32, mdmp_register_bcast_i64
  end interface

contains

  ! ------------------------------------------------------------------
  ! Simple Wrappers
  ! ------------------------------------------------------------------
  subroutine mdmp_comm_init()
    call c_mdmp_init()
  end subroutine
  subroutine mdmp_comm_final()
    call c_mdmp_final()
  end subroutine
  integer function mdmp_get_size()
    mdmp_get_size = c_mdmp_get_size()
  end function
  integer function mdmp_get_rank()
    mdmp_get_rank = c_mdmp_get_rank()
  end function
  subroutine mdmp_commregion_begin()
    call c_mdmp_commregion_begin()
  end subroutine
  subroutine mdmp_commregion_end()
    call c_mdmp_commregion_end()
  end subroutine
  subroutine mdmp_comm_sync()
    call c_mdmp_sync()
  end subroutine
  real(c_double) function mdmp_wtime()
    mdmp_wtime = c_mdmp_wtime()
  end function
  subroutine mdmp_set_debug(enable)
    integer, intent(in) :: enable
    call c_mdmp_set_debug(int(enable, c_int))
  end subroutine
  subroutine mdmp_abort(err_code)
    integer, intent(in) :: err_code
    call c_mdmp_abort(int(err_code, c_int))
  end subroutine
  integer function mdmp_commit()
    mdmp_commit = c_mdmp_commit()
  end function

  ! ------------------------------------------------------------------
  ! Double Precision (Real64) Implementations
  ! ------------------------------------------------------------------
  integer function mdmp_send_r64(buf, count, sender, dest, tag)
    real(c_double), intent(inout) :: buf(*)
    integer, intent(in) :: count, sender, dest, tag
    mdmp_send_r64 = c_mdmp_send(buf, int(count, c_size_t), MDMP_TYPE_DOUBLE, int(count, c_size_t)*c_sizeof(buf(1)), int(sender, c_int), int(dest, c_int), int(tag, c_int))
  end function
  integer function mdmp_recv_r64(buf, count, receiver, src, tag)
    real(c_double), intent(inout) :: buf(*)
    integer, intent(in) :: count, receiver, src, tag
    mdmp_recv_r64 = c_mdmp_recv(buf, int(count, c_size_t), MDMP_TYPE_DOUBLE, int(count, c_size_t)*c_sizeof(buf(1)), int(receiver, c_int), int(src, c_int), int(tag, c_int))
  end function
  integer function mdmp_reduce_r64(in_buf, out_buf, count, root, op)
    real(c_double), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, root, op
    mdmp_reduce_r64 = c_mdmp_reduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_DOUBLE, int(count, c_size_t)*c_sizeof(in_buf(1)), int(root, c_int), int(op, c_int))
  end function
  integer function mdmp_gather_r64(send_buf, send_count, recv_buf, root)
    real(c_double), intent(inout) :: send_buf(*), recv_buf(*)
    integer, intent(in) :: send_count, root
    mdmp_gather_r64 = c_mdmp_gather(send_buf, int(send_count, c_size_t), recv_buf, MDMP_TYPE_DOUBLE, int(send_count, c_size_t)*c_sizeof(send_buf(1)), int(root, c_int))
  end function
  integer function mdmp_allreduce_r64(in_buf, out_buf, count, op)
    real(c_double), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, op
    mdmp_allreduce_r64 = c_mdmp_allreduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_DOUBLE, int(count, c_size_t)*c_sizeof(in_buf(1)), int(op, c_int))
  end function
  integer function mdmp_allgather_r64(in_buf, count, out_buf)
    real(c_double), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count
    mdmp_allgather_r64 = c_mdmp_allgather(in_buf, int(count, c_size_t), out_buf, MDMP_TYPE_DOUBLE, int(count, c_size_t)*c_sizeof(in_buf(1)))
  end function
  integer function mdmp_bcast_r64(buf, count, root)
    real(c_double), intent(inout) :: buf(*)
    integer, intent(in) :: count, root
    mdmp_bcast_r64 = c_mdmp_bcast(buf, int(count, c_size_t), MDMP_TYPE_DOUBLE, int(count, c_size_t)*c_sizeof(buf(1)), int(root, c_int))
  end function

  ! --- Declarative Real64 ---
  subroutine mdmp_register_send_r64(buf, count, sender, dest, tag)
    real(c_double), intent(inout) :: buf(*)
    integer, intent(in) :: count, sender, dest, tag
    call c_mdmp_register_send(buf, int(count, c_size_t), MDMP_TYPE_DOUBLE, int(count, c_size_t)*c_sizeof(buf(1)), int(sender, c_int), int(dest, c_int), int(tag, c_int))
  end subroutine
  subroutine mdmp_register_recv_r64(buf, count, receiver, src, tag)
    real(c_double), intent(inout) :: buf(*)
    integer, intent(in) :: count, receiver, src, tag
    call c_mdmp_register_recv(buf, int(count, c_size_t), MDMP_TYPE_DOUBLE, int(count, c_size_t)*c_sizeof(buf(1)), int(receiver, c_int), int(src, c_int), int(tag, c_int))
  end subroutine
  subroutine mdmp_register_reduce_r64(in_buf, out_buf, count, root, op)
    real(c_double), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, root, op
    call c_mdmp_register_reduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_DOUBLE, int(count, c_size_t)*c_sizeof(in_buf(1)), int(root, c_int), int(op, c_int))
  end subroutine
  subroutine mdmp_register_gather_r64(send_buf, send_count, recv_buf, root)
    real(c_double), intent(inout) :: send_buf(*), recv_buf(*)
    integer, intent(in) :: send_count, root
    call c_mdmp_register_gather(send_buf, int(send_count, c_size_t), recv_buf, MDMP_TYPE_DOUBLE, int(send_count, c_size_t)*c_sizeof(send_buf(1)), int(root, c_int))
  end subroutine
  subroutine mdmp_register_allreduce_r64(in_buf, out_buf, count, op)
    real(c_double), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, op
    call c_mdmp_register_allreduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_DOUBLE, int(count, c_size_t)*c_sizeof(in_buf(1)), int(op, c_int))
  end subroutine
  subroutine mdmp_register_allgather_r64(in_buf, count, out_buf)
    real(c_double), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count
    call c_mdmp_register_allgather(in_buf, int(count, c_size_t), out_buf, MDMP_TYPE_DOUBLE, int(count, c_size_t)*c_sizeof(in_buf(1)))
  end subroutine
  subroutine mdmp_register_bcast_r64(buf, count, root)
    real(c_double), intent(inout) :: buf(*)
    integer, intent(in) :: count, root
    call c_mdmp_register_bcast(buf, int(count, c_size_t), MDMP_TYPE_DOUBLE, int(count, c_size_t)*c_sizeof(buf(1)), int(root, c_int))
  end subroutine

  ! ------------------------------------------------------------------
  ! Single Precision (Real32) Implementations
  ! ------------------------------------------------------------------
  integer function mdmp_send_r32(buf, count, sender, dest, tag)
    real(c_float), intent(inout) :: buf(*)
    integer, intent(in) :: count, sender, dest, tag
    mdmp_send_r32 = c_mdmp_send(buf, int(count, c_size_t), MDMP_TYPE_FLOAT, int(count, c_size_t)*c_sizeof(buf(1)), int(sender, c_int), int(dest, c_int), int(tag, c_int))
  end function
  integer function mdmp_recv_r32(buf, count, receiver, src, tag)
    real(c_float), intent(inout) :: buf(*)
    integer, intent(in) :: count, receiver, src, tag
    mdmp_recv_r32 = c_mdmp_recv(buf, int(count, c_size_t), MDMP_TYPE_FLOAT, int(count, c_size_t)*c_sizeof(buf(1)), int(receiver, c_int), int(src, c_int), int(tag, c_int))
  end function
  integer function mdmp_reduce_r32(in_buf, out_buf, count, root, op)
    real(c_float), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, root, op
    mdmp_reduce_r32 = c_mdmp_reduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_FLOAT, int(count, c_size_t)*c_sizeof(in_buf(1)), int(root, c_int), int(op, c_int))
  end function
  integer function mdmp_gather_r32(send_buf, send_count, recv_buf, root)
    real(c_float), intent(inout) :: send_buf(*), recv_buf(*)
    integer, intent(in) :: send_count, root
    mdmp_gather_r32 = c_mdmp_gather(send_buf, int(send_count, c_size_t), recv_buf, MDMP_TYPE_FLOAT, int(send_count, c_size_t)*c_sizeof(send_buf(1)), int(root, c_int))
  end function
  integer function mdmp_allreduce_r32(in_buf, out_buf, count, op)
    real(c_float), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, op
    mdmp_allreduce_r32 = c_mdmp_allreduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_FLOAT, int(count, c_size_t)*c_sizeof(in_buf(1)), int(op, c_int))
  end function
  integer function mdmp_allgather_r32(in_buf, count, out_buf)
    real(c_float), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count
    mdmp_allgather_r32 = c_mdmp_allgather(in_buf, int(count, c_size_t), out_buf, MDMP_TYPE_FLOAT, int(count, c_size_t)*c_sizeof(in_buf(1)))
  end function
  integer function mdmp_bcast_r32(buf, count, root)
    real(c_float), intent(inout) :: buf(*)
    integer, intent(in) :: count, root
    mdmp_bcast_r32 = c_mdmp_bcast(buf, int(count, c_size_t), MDMP_TYPE_FLOAT, int(count, c_size_t)*c_sizeof(buf(1)), int(root, c_int))
  end function

  ! --- Declarative Real32 ---
  subroutine mdmp_register_send_r32(buf, count, sender, dest, tag)
    real(c_float), intent(inout) :: buf(*)
    integer, intent(in) :: count, sender, dest, tag
    call c_mdmp_register_send(buf, int(count, c_size_t), MDMP_TYPE_FLOAT, int(count, c_size_t)*c_sizeof(buf(1)), int(sender, c_int), int(dest, c_int), int(tag, c_int))
  end subroutine
  subroutine mdmp_register_recv_r32(buf, count, receiver, src, tag)
    real(c_float), intent(inout) :: buf(*)
    integer, intent(in) :: count, receiver, src, tag
    call c_mdmp_register_recv(buf, int(count, c_size_t), MDMP_TYPE_FLOAT, int(count, c_size_t)*c_sizeof(buf(1)), int(receiver, c_int), int(src, c_int), int(tag, c_int))
  end subroutine
  subroutine mdmp_register_reduce_r32(in_buf, out_buf, count, root, op)
    real(c_float), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, root, op
    call c_mdmp_register_reduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_FLOAT, int(count, c_size_t)*c_sizeof(in_buf(1)), int(root, c_int), int(op, c_int))
  end subroutine
  subroutine mdmp_register_gather_r32(send_buf, send_count, recv_buf, root)
    real(c_float), intent(inout) :: send_buf(*), recv_buf(*)
    integer, intent(in) :: send_count, root
    call c_mdmp_register_gather(send_buf, int(send_count, c_size_t), recv_buf, MDMP_TYPE_FLOAT, int(send_count, c_size_t)*c_sizeof(send_buf(1)), int(root, c_int))
  end subroutine
  subroutine mdmp_register_allreduce_r32(in_buf, out_buf, count, op)
    real(c_float), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, op
    call c_mdmp_register_allreduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_FLOAT, int(count, c_size_t)*c_sizeof(in_buf(1)), int(op, c_int))
  end subroutine
  subroutine mdmp_register_allgather_r32(in_buf, count, out_buf)
    real(c_float), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count
    call c_mdmp_register_allgather(in_buf, int(count, c_size_t), out_buf, MDMP_TYPE_FLOAT, int(count, c_size_t)*c_sizeof(in_buf(1)))
  end subroutine
  subroutine mdmp_register_bcast_r32(buf, count, root)
    real(c_float), intent(inout) :: buf(*)
    integer, intent(in) :: count, root
    call c_mdmp_register_bcast(buf, int(count, c_size_t), MDMP_TYPE_FLOAT, int(count, c_size_t)*c_sizeof(buf(1)), int(root, c_int))
  end subroutine
  
  ! ------------------------------------------------------------------
  ! Integer (Int32) Implementations
  ! ------------------------------------------------------------------
  integer function mdmp_send_i32(buf, count, sender, dest, tag)
    integer(c_int), intent(inout) :: buf(*)
    integer, intent(in) :: count, sender, dest, tag
    mdmp_send_i32 = c_mdmp_send(buf, int(count, c_size_t), MDMP_TYPE_INT, int(count, c_size_t)*c_sizeof(buf(1)), int(sender, c_int), int(dest, c_int), int(tag, c_int))
  end function
  integer function mdmp_recv_i32(buf, count, receiver, src, tag)
    integer(c_int), intent(inout) :: buf(*)
    integer, intent(in) :: count, receiver, src, tag
    mdmp_recv_i32 = c_mdmp_recv(buf, int(count, c_size_t), MDMP_TYPE_INT, int(count, c_size_t)*c_sizeof(buf(1)), int(receiver, c_int), int(src, c_int), int(tag, c_int))
  end function
  integer function mdmp_reduce_i32(in_buf, out_buf, count, root, op)
    integer(c_int), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, root, op
    mdmp_reduce_i32 = c_mdmp_reduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_INT, int(count, c_size_t)*c_sizeof(in_buf(1)), int(root, c_int), int(op, c_int))
  end function
  integer function mdmp_gather_i32(send_buf, send_count, recv_buf, root)
    integer(c_int), intent(inout) :: send_buf(*), recv_buf(*)
    integer, intent(in) :: send_count, root
    mdmp_gather_i32 = c_mdmp_gather(send_buf, int(send_count, c_size_t), recv_buf, MDMP_TYPE_INT, int(send_count, c_size_t)*c_sizeof(send_buf(1)), int(root, c_int))
  end function
  integer function mdmp_allreduce_i32(in_buf, out_buf, count, op)
    integer(c_int), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, op
    mdmp_allreduce_i32 = c_mdmp_allreduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_INT, int(count, c_size_t)*c_sizeof(in_buf(1)), int(op, c_int))
  end function
  integer function mdmp_allgather_i32(in_buf, count, out_buf)
    integer(c_int), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count
    mdmp_allgather_i32 = c_mdmp_allgather(in_buf, int(count, c_size_t), out_buf, MDMP_TYPE_INT, int(count, c_size_t)*c_sizeof(in_buf(1)))
  end function
  integer function mdmp_bcast_i32(buf, count, root)
    integer(c_int), intent(inout) :: buf(*)
    integer, intent(in) :: count, root
    mdmp_bcast_i32 = c_mdmp_bcast(buf, int(count, c_size_t), MDMP_TYPE_INT, int(count, c_size_t)*c_sizeof(buf(1)), int(root, c_int))
  end function

  ! --- Declarative Int32 ---
  subroutine mdmp_register_send_i32(buf, count, sender, dest, tag)
    integer(c_int), intent(inout) :: buf(*)
    integer, intent(in) :: count, sender, dest, tag
    call c_mdmp_register_send(buf, int(count, c_size_t), MDMP_TYPE_INT, int(count, c_size_t)*c_sizeof(buf(1)), int(sender, c_int), int(dest, c_int), int(tag, c_int))
  end subroutine
  subroutine mdmp_register_recv_i32(buf, count, receiver, src, tag)
    integer(c_int), intent(inout) :: buf(*)
    integer, intent(in) :: count, receiver, src, tag
    call c_mdmp_register_recv(buf, int(count, c_size_t), MDMP_TYPE_INT, int(count, c_size_t)*c_sizeof(buf(1)), int(receiver, c_int), int(src, c_int), int(tag, c_int))
  end subroutine
  subroutine mdmp_register_reduce_i32(in_buf, out_buf, count, root, op)
    integer(c_int), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, root, op
    call c_mdmp_register_reduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_INT, int(count, c_size_t)*c_sizeof(in_buf(1)), int(root, c_int), int(op, c_int))
  end subroutine
  subroutine mdmp_register_gather_i32(send_buf, send_count, recv_buf, root)
    integer(c_int), intent(inout) :: send_buf(*), recv_buf(*)
    integer, intent(in) :: send_count, root
    call c_mdmp_register_gather(send_buf, int(send_count, c_size_t), recv_buf, MDMP_TYPE_INT, int(send_count, c_size_t)*c_sizeof(send_buf(1)), int(root, c_int))
  end subroutine
  subroutine mdmp_register_allreduce_i32(in_buf, out_buf, count, op)
    integer(c_int), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, op
    call c_mdmp_register_allreduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_INT, int(count, c_size_t)*c_sizeof(in_buf(1)), int(op, c_int))
  end subroutine
  subroutine mdmp_register_allgather_i32(in_buf, count, out_buf)
    integer(c_int), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count
    call c_mdmp_register_allgather(in_buf, int(count, c_size_t), out_buf, MDMP_TYPE_INT, int(count, c_size_t)*c_sizeof(in_buf(1)))
  end subroutine
  subroutine mdmp_register_bcast_i32(buf, count, root)
    integer(c_int), intent(inout) :: buf(*)
    integer, intent(in) :: count, root
    call c_mdmp_register_bcast(buf, int(count, c_size_t), MDMP_TYPE_INT, int(count, c_size_t)*c_sizeof(buf(1)), int(root, c_int))
  end subroutine

  ! ------------------------------------------------------------------
  ! Integer (Int64) Implementations
  ! ------------------------------------------------------------------
  integer function mdmp_send_i64(buf, count, sender, dest, tag)
    integer(c_int64_t), intent(inout) :: buf(*)
    integer, intent(in) :: count, sender, dest, tag
    mdmp_send_i64 = c_mdmp_send(buf, int(count, c_size_t), MDMP_TYPE_INT64, int(count, c_size_t)*c_sizeof(buf(1)), int(sender, c_int), int(dest, c_int), int(tag, c_int))
  end function
  integer function mdmp_recv_i64(buf, count, receiver, src, tag)
    integer(c_int64_t), intent(inout) :: buf(*)
    integer, intent(in) :: count, receiver, src, tag
    mdmp_recv_i64 = c_mdmp_recv(buf, int(count, c_size_t), MDMP_TYPE_INT64, int(count, c_size_t)*c_sizeof(buf(1)), int(receiver, c_int), int(src, c_int), int(tag, c_int))
  end function
  integer function mdmp_reduce_i64(in_buf, out_buf, count, root, op)
    integer(c_int64_t), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, root, op
    mdmp_reduce_i64 = c_mdmp_reduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_INT64, int(count, c_size_t)*c_sizeof(in_buf(1)), int(root, c_int), int(op, c_int))
  end function
  integer function mdmp_gather_i64(send_buf, send_count, recv_buf, root)
    integer(c_int64_t), intent(inout) :: send_buf(*), recv_buf(*)
    integer, intent(in) :: send_count, root
    mdmp_gather_i64 = c_mdmp_gather(send_buf, int(send_count, c_size_t), recv_buf, MDMP_TYPE_INT64, int(send_count, c_size_t)*c_sizeof(send_buf(1)), int(root, c_int))
  end function
  integer function mdmp_allreduce_i64(in_buf, out_buf, count, op)
    integer(c_int64_t), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, op
    mdmp_allreduce_i64 = c_mdmp_allreduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_INT64, int(count, c_size_t)*c_sizeof(in_buf(1)), int(op, c_int))
  end function
  integer function mdmp_allgather_i64(in_buf, count, out_buf)
    integer(c_int64_t), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count
    mdmp_allgather_i64 = c_mdmp_allgather(in_buf, int(count, c_size_t), out_buf, MDMP_TYPE_INT64, int(count, c_size_t)*c_sizeof(in_buf(1)))
  end function
  integer function mdmp_bcast_i64(buf, count, root)
    integer(c_int64_t), intent(inout) :: buf(*)
    integer, intent(in) :: count, root
    mdmp_bcast_i64 = c_mdmp_bcast(buf, int(count, c_size_t), MDMP_TYPE_INT64, int(count, c_size_t)*c_sizeof(buf(1)), int(root, c_int))
  end function

  ! --- Declarative Int64 ---
  subroutine mdmp_register_send_i64(buf, count, sender, dest, tag)
    integer(c_int64_t), intent(inout) :: buf(*)
    integer, intent(in) :: count, sender, dest, tag
    call c_mdmp_register_send(buf, int(count, c_size_t), MDMP_TYPE_INT64, int(count, c_size_t)*c_sizeof(buf(1)), int(sender, c_int), int(dest, c_int), int(tag, c_int))
  end subroutine
  subroutine mdmp_register_recv_i64(buf, count, receiver, src, tag)
    integer(c_int64_t), intent(inout) :: buf(*)
    integer, intent(in) :: count, receiver, src, tag
    call c_mdmp_register_recv(buf, int(count, c_size_t), MDMP_TYPE_INT64, int(count, c_size_t)*c_sizeof(buf(1)), int(receiver, c_int), int(src, c_int), int(tag, c_int))
  end subroutine
  subroutine mdmp_register_reduce_i64(in_buf, out_buf, count, root, op)
    integer(c_int64_t), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, root, op
    call c_mdmp_register_reduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_INT64, int(count, c_size_t)*c_sizeof(in_buf(1)), int(root, c_int), int(op, c_int))
  end subroutine
  subroutine mdmp_register_gather_i64(send_buf, send_count, recv_buf, root)
    integer(c_int64_t), intent(inout) :: send_buf(*), recv_buf(*)
    integer, intent(in) :: send_count, root
    call c_mdmp_register_gather(send_buf, int(send_count, c_size_t), recv_buf, MDMP_TYPE_INT64, int(send_count, c_size_t)*c_sizeof(send_buf(1)), int(root, c_int))
  end subroutine
  subroutine mdmp_register_allreduce_i64(in_buf, out_buf, count, op)
    integer(c_int64_t), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count, op
    call c_mdmp_register_allreduce(in_buf, out_buf, int(count, c_size_t), MDMP_TYPE_INT64, int(count, c_size_t)*c_sizeof(in_buf(1)), int(op, c_int))
  end subroutine
  subroutine mdmp_register_allgather_i64(in_buf, count, out_buf)
    integer(c_int64_t), intent(inout) :: in_buf(*), out_buf(*)
    integer, intent(in) :: count
    call c_mdmp_register_allgather(in_buf, int(count, c_size_t), out_buf, MDMP_TYPE_INT64, int(count, c_size_t)*c_sizeof(in_buf(1)))
  end subroutine
  subroutine mdmp_register_bcast_i64(buf, count, root)
    integer(c_int64_t), intent(inout) :: buf(*)
    integer, intent(in) :: count, root
    call c_mdmp_register_bcast(buf, int(count, c_size_t), MDMP_TYPE_INT64, int(count, c_size_t)*c_sizeof(buf(1)), int(root, c_int))
  end subroutine
  
end module mdmp_interface

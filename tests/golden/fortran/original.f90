module sample_mod
  implicit none
contains
  function add(a, b) result(c)
    integer, intent(in) :: a, b
    integer :: c
    c = a + b
  end function add
end module sample_mod

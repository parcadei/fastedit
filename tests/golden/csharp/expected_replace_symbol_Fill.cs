public class Cart
{
    private const int Capacity = 4;

    public static int Add(int items)
    {
        return items + 1;
    }

    public static int Fill(int items)
    {
        return items * 3;
    }
}

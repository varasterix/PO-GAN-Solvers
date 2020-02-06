public class Point {

    private int x;
    private int y;

    public Point(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public Point() {
        this.x = (int) (Math.random() * 1000);
        this.y = (int) (Math.random() * 1000);
    }

    public int getX() {
        return this.x;
    }

    public int getY() {
        return this.y;
    }

    public int dist(Point pt) {
        return (int) Math.sqrt(Math.pow((this.getX() - pt.getX()), 2) + Math.pow((this.getY() - pt.getY()), 2));
    }

    @Override
    public String toString() {
        String res = this.getX() + "\t" + this.getY();
        return res;
    }

    public static void main(String[] args) {
        String res = "";
        Point a = new Point();
        res = a.toString();
        System.out.println(res);
    }

}

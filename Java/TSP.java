public class TSP {

    public static int nb_cities;
    public static int[][] matrix;
    public static Point[] points;

    public TSP(int nb_cities) {
        this.nb_cities = nb_cities;
        this.points = new Point[this.nb_cities];
        for (int i=0; i<this.nb_cities; i++) {
            this.points[i] = new Point();
        }
        this.matrix = new int[this.nb_cities][this.nb_cities];
        for (int j=0; j<this.nb_cities; j++) {
            for (int k=0; k<j; k++) {
                this.matrix[j][k] = points[j].dist(points[k]);
                this.matrix[k][j] = this.matrix[j][k];
            }
        }
    }

    public int getNb_cities() {
        return this.nb_cities;
    }

    public Point[] getPoints() {
        return this.points;
    }

    public int[][] getMatrix() {
        return this.matrix;
    }

    @Override
    public String toString() {
        String res = "";
        for (int i=0; i<this.getMatrix().length; i++) {
            for (int j=0; j<this.getMatrix().length; j++) {
                res += this.getMatrix()[i][j];
                res += (j<this.getMatrix().length-1) ? "\t" : ((i<this.getMatrix().length-1) ? "\n" : "");
            }
        }
        return res;
    }
}

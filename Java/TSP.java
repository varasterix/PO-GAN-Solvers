public class TSP {

    public static int nb_cities;
    public static int[][] matrix;

    public TSP(int nb_cities) {
        this.nb_cities = nb_cities;
        this.matrix = new int[this.nb_cities][this.nb_cities];
        for (int i=0; i<this.nb_cities; i++) {
            for (int j=0; j<nb_cities; j++) {
                this.matrix[i][j] = (int)(Math.random() * 1000);
            }
        }
    }

    public int getNb_cities() {
        return this.nb_cities;
    }

    public int[][] getMatrix() {
        return this.matrix;
    }

    @Override
    public String toString() {
        String res = "[";
        for (int i=0; i<this.getNb_cities(); i++) {
            for (int j=0; j<this.getNb_cities() - 1; j++) {
                res += this.getMatrix()[i][j] + ", ";
            }
        }
        res += this.getMatrix()[this.getNb_cities() - 1][this.getNb_cities() - 1] + "]";
        return res;
    }

    public static void main(String[] args) {
        int nb_cities = 10;
        TSP tsp = new TSP(nb_cities);
        System.out.println(tsp.toString());
    }
}

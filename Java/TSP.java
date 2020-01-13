public class TSP {

    public static int nb_cities;
    public static int[][] matrix;

    public TSP(int nb_cities) {
        this.nb_cities = nb_cities;
        this.matrix = new int[this.nb_cities][this.nb_cities];
        for (int i = 0; i < this.nb_cities; i++) {
            for (int j = 0; j < nb_cities; j++) {
                this.matrix[i][j] = (int) (Math.random() * 1000);
            }
            this.matrix[i][i] = 0;
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
        String res = "";
        for (int i = 0; i < this.getMatrix().length - 1; i++) {
            for (int j = 0; j < this.getMatrix().length; j++) {
                res += this.getMatrix()[i][j] + "\t";
            }
            res += "\n";
        }
        for (int j = 0; j < this.getMatrix().length; j++) {
            res += this.getMatrix()[this.getNb_cities() - 1][j] + "\t";
        }
        return res;
    }
}

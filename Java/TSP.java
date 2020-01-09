import java.io.*;

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
        String res = "";
        for (int i=0; i<this.getMatrix().length; i++) {
            for (int j=0; j<this.getMatrix().length; j++) {
                res += this.getMatrix()[i][j] + " ";
            }
            res += "\n";
        }
        return res;
    }

    public static void main(String[] args) throws UnsupportedEncodingException, FileNotFoundException {
        int nb_cities = 10;

        for (int i=0; i<2000; i++) {
            TSP tsp = new TSP(nb_cities);

            PrintWriter writer = new PrintWriter(String.format("dataSet_%d_%d.tsp", tsp.getNb_cities(), i), "UTF-8");
            writer.println(i);
            writer.println(tsp.getNb_cities());
            writer.println(tsp.toString());
            writer.close();
        }
    }
}

import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solution;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.variables.IntVar;

import java.io.*;

public class Main {

    public static final int NB_TSP = 10; // Number of instances to create
    public static final int NB_CITIES = 25; // Number of cities for each instance

    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {

        // Create the model
        for (int i=0; i<NB_TSP; i++) { // Loop for all instances
            Model model = new Model();
            Solver solver = model.getSolver();

            // Declare the variables
            TSP tsp = new TSP(NB_CITIES);
            IntVar[] successors = model.intVarArray(NB_CITIES, 0, NB_CITIES);

            // Declare and post the constraints
            model.allDifferent(successors).post();
            model.subCircuit(successors, 0, model.intVar(NB_CITIES)).post();

            IntVar[] distances = model.intVarArray(NB_CITIES, 0, 9999);
            for (int j=0; j<NB_CITIES; j++) {model.element(distances[j], tsp.getMatrix()[j], successors[j]).post();}
            IntVar z = model.intVar(0, 9999);
            model.sum(distances, "=", z).post();

            // Search the best solution
            solver.showSolutions();
//            solver.showShortStatistics();
            Solution sol = solver.findOptimalSolution(z, false);

            // Print solution to a .txt file
            String res = "";
            int[] ordered_path = new int[NB_CITIES];
            ordered_path[0] = successors[0].getValue();
            for (int k=1; k<NB_CITIES; k++) {ordered_path[k] = sol.getIntVal(successors[ordered_path[k-1]]);}

        for (int k=0; k<ordered_path.length; k++) {
                res += ordered_path[k];
                res += (k<ordered_path.length-1) ? "\t" : "";
            }
            PrintWriter writer = new PrintWriter(
                    String.format("dataSet_%d_%d.choco", tsp.getNb_cities(), i), "UTF-8");
            writer.println(i);
            writer.println(tsp.getNb_cities());
            writer.println(tsp.toString());
            writer.println(res);
            writer.println(sol.getIntVal(z));
            for (int k=0; k<tsp.getNb_cities(); k++) writer.println(tsp.getPoints()[k]);
            writer.close();
        }
    }
}

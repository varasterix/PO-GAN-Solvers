import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.variables.IntVar;

public class Main {

    public static final int NB_TSP = 2000; // Number of instances to create
    public static final int NB_CITIES = 10; // Number of cities for each instance

    public static void main(String[] args) {

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
            solver.findOptimalSolution(z, false);
        }
    }

}

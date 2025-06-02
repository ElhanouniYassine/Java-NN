import java.util.ArrayList;
import java.util.Random;

public class NN {
    Matrix weights_input_to_hidden;
    Matrix weights_hidden_to_output;
    Matrix bias;
    double bias_hidden_to_output;
    Matrix input;
    Matrix tempInput;
    Matrix MatrixBigOutput;
    double[][] bigInput;
    double[][] bigOutput;

    public NN() {
        bigInput= new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        bigOutput = new double[][]{{1}, {0}, {0}, {1}};
        MatrixBigOutput = new Matrix(4, 1);
        for (int i = 0; i < 4; i++) {
            MatrixBigOutput.set(i, 0, bigOutput[i][0]);
        }

        weights_input_to_hidden = new Matrix(2, 2);
        weights_input_to_hidden.fillRandom(-1, 1);

        weights_hidden_to_output=new Matrix(2, 1);
        weights_hidden_to_output.fillRandom(-1, 1);

        bias_hidden_to_output = new Random().nextDouble() * 2 - 1;

        bias = new Matrix(2, 1);
        bias.fillRandom(-1, 1);

        input = new Matrix(2, 1);
        input.set(0, 0, 1.0);
        input.set(1, 0, 0.0);
    }



    public ArrayList<Matrix> forward() {
        ArrayList<Matrix> result = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            tempInput = new Matrix(2, 1);
            tempInput.set(0, 0, bigInput[i][0]);
            tempInput.set(1, 0, bigInput[i][1]);
            Matrix hiddenActivation = weights_input_to_hidden.multiply(tempInput).sum(bias).sigmoid();
            result.add(hiddenActivation);
        }
        return result;
    }


    public ArrayList<Matrix> nextForward(){
        ArrayList<Matrix> A1= forward();
        ArrayList<Matrix> result=new ArrayList<>();
        for(int i=0; i<4; i++){
            // hadi hya Z2=weights_hidden_to_output.transpose().multiply(A1.get(i)).scalarSum(bias_hidden_to_output)
            result.add(weights_hidden_to_output.transpose().multiply(A1.get(i)).scalarSum(bias_hidden_to_output).sigmoid());
        }
        return result;
    }

    public ArrayList<Matrix> outputErrors() {
        ArrayList<Matrix> A2 = nextForward();
        ArrayList<Matrix> result = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            Matrix target = new Matrix(1, 1);
            target.set(0, 0, MatrixBigOutput.get(i, 0));
            result.add(A2.get(i).subtract(target));
        }
        return result;
    }

    public ArrayList<Matrix> computeDelta2() {
        ArrayList<Matrix> result = new ArrayList<>();
        ArrayList<Matrix> errors = outputErrors();         // (A2 - Y)
        ArrayList<Matrix> A2 = nextForward();              // predicted outputs

        for (int i = 0; i < 4; i++) {
            Matrix sigmoidDeriv = A2.get(i).sigmoidDerivative();  // A2 * (1 - A2)
            Matrix delta2 = sigmoidDeriv.hadamard(errors.get(i)); // δ2 = (A2 - Y) ⊙ σ'(A2)
            result.add(delta2);
        }

        return result;
    }


    public ArrayList<Matrix> computeDelta1() {
        ArrayList<Matrix> result = new ArrayList<>();
        ArrayList<Matrix> delta2List = computeDelta2();

        for (int i = 0; i < 4; i++) {

            Matrix tempInput = new Matrix(2, 1);
            tempInput.set(0, 0, bigInput[i][0]);
            tempInput.set(1, 0, bigInput[i][1]);

            Matrix z1 = weights_input_to_hidden.multiply(tempInput).sum(bias);
            Matrix a1 = z1.sigmoid();
            Matrix sigmoidDerivative = a1.sigmoidDerivative();

            Matrix delta2 = delta2List.get(i);
            Matrix propagatedError = weights_hidden_to_output.multiply(delta2);
            Matrix delta1 = propagatedError.hadamard(sigmoidDerivative);
            result.add(delta1);
        }

        return result;
    }



    public void train(int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < 4; i++) {
                Matrix X = new Matrix(2, 1);
                X.set(0, 0, bigInput[i][0]);
                X.set(1, 0, bigInput[i][1]);

                double y = bigOutput[i][0];
                Matrix target = new Matrix(1, 1);
                target.set(0, 0, y);

                Matrix Z1 = weights_input_to_hidden.multiply(X).sum(bias);
                Matrix A1 = Z1.sigmoid();

                Matrix Z2 = weights_hidden_to_output.transpose().multiply(A1).scalarSum(bias_hidden_to_output);
                Matrix A2 = Z2.sigmoid();

                Matrix error = A2.subtract(target);
                Matrix delta2 = error.hadamard(A2.sigmoidDerivative());

                Matrix hiddenError = weights_hidden_to_output.multiply(delta2);
                Matrix delta1 = hiddenError.hadamard(A1.sigmoidDerivative());

                Matrix grad_W2 = A1.multiply(delta2.transpose());
                Matrix grad_W1 = X.multiply(delta1.transpose());
                weights_hidden_to_output = weights_hidden_to_output.subtract(grad_W2.scalarMultiply(learningRate));
                weights_input_to_hidden = weights_input_to_hidden.subtract(grad_W1.scalarMultiply(learningRate));

                bias_hidden_to_output -= learningRate * delta2.get(0, 0);

                bias = bias.subtract(delta1.scalarMultiply(learningRate));
            }
        }
    }



    public void log() {
        System.out.println("Weights:");
        weights_input_to_hidden.print();

        System.out.println("Bias:");
        bias.print();



        System.out.println("Output:");
        ArrayList<Matrix> result=nextForward();
        for(Matrix m:result){
            m.print();
        }
    }
}

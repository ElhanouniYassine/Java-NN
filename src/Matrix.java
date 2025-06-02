import java.util.Random;

public class Matrix {
    private double[][] data;
    private int rows, cols;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public double get(int row, int col) {
        return data[row][col];
    }

    public void set(int row, int col, double value) {
        data[row][col] = value;
    }

    public void fillRandom(double min, double max) {
        Random rand = new Random();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                data[i][j] = min + (max - min) * rand.nextDouble();
    }

    public double[][] getData() {
        return data;
    }

    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Matrix dimensions do not match");
        }

        Matrix result = new Matrix(this.rows, other.cols);

        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                result.set(i, j, sum);
            }
        }

        return result;
    }

    public Matrix sum(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols ) {
            throw new IllegalArgumentException("Matrix can't be added");
        }
        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0;
                sum+=this.data[i][j]+other.data[i][j];
                result.set(i, j, sum);
            }

        }
        return result;
    }

    public Matrix sigmoid(){
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.set(i, j, 1 / (1 + Math.exp(-data[i][j])));
            }
        }
        return result;
    }
    public Matrix transpose() {
        Matrix result = new Matrix(this.cols, this.rows);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.set(j, i, this.data[i][j]);
            }
        }
        return result;
    }

    public Matrix scalarMultiply(double scalar) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.set(i, j, this.get(i,j) * scalar);
            }
        }
        return result;
    }
    public Matrix scalarSum(double scalar) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.set(i, j, this.get(i,j) + scalar);
            }
        }
        return result;
    }
    public Matrix subtract(Matrix other)
    {
        if(this.rows != other.rows || this.cols != other.cols ){
            throw new IllegalArgumentException("Matrix dimensions do not match");
        }
        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                result.set(i, j, this.get(i,j) - other.get(i,j));
            }
        }
        return result;
    }

    public Matrix hadamard(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for Hadamard product.");
        }

        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.set(i, j, this.get(i, j) * other.get(i, j));
            }
        }
        return result;
    }


    public Matrix sigmoidDerivative() {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double val = this.data[i][j]; // hadi aslan sigmoid(x)
                result.set(i, j, val * (1 - val));
            }
        }
        return result;
    }




    public void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.printf("%.4f ", data[i][j]);
            }
            System.out.println();
        }
    }

}

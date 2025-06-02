public class Main {
    public static void main(String[] args) {
        NN nn = new NN();
        System.out.println("Before training:");
        nn.log();

        nn.train(500000, 0.05);

        System.out.println("\nAfter training:");
        nn.log();
    }
}

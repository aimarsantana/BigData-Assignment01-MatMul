import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Basic dense matrix multiplication C = A x B (triple loop, i-j-k order).
 *
 * Storage: double[][] (array of row arrays; each row is a contiguous double[],
 * rows are separate heap objects). Numeric type: double (IEEE-754 binary64).
 *
 * Usage: java -cp build MatMul A.bin B.bin n warmup reps batch [C_out.bin]
 *
 * The timed region contains ONLY the triple loop. Timer: System.nanoTime().
 * Warm-up samples (JIT compilation) are printed with phase=warmup and are
 * excluded from the reported statistics.
 */
public class MatMul {

    static double[][] readMatrix(String path, int n) throws IOException {
        byte[] bytes = Files.readAllBytes(Paths.get(path));
        if (bytes.length != 8L * n * n) {
            throw new IOException("Unexpected size for " + path + ": " + bytes.length);
        }
        ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i][j] = bb.getDouble();
            }
        }
        return m;
    }

    static void writeMatrix(String path, double[][] m) throws IOException {
        int n = m.length;
        ByteBuffer bb = ByteBuffer.allocate(8 * n).order(ByteOrder.LITTLE_ENDIAN);
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))) {
            for (double[] row : m) {
                bb.clear();
                for (double v : row) bb.putDouble(v);
                out.write(bb.array());
            }
        }
    }

    /** Straightforward O(n^3) triple loop, i-j-k order, scalar accumulator. */
    static void matmul(double[][] A, double[][] B, double[][] C, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += A[i][k] * B[k][j];
                }
                C[i][j] = s;
            }
        }
    }

    /** Peak resident set size (VmHWM) of this process, from /proc/self/status (Linux). */
    static long peakRssKib() {
        try {
            for (String line : Files.readAllLines(Paths.get("/proc/self/status"))) {
                if (line.startsWith("VmHWM:")) {
                    return Long.parseLong(line.replaceAll("[^0-9]", ""));
                }
            }
        } catch (IOException | NumberFormatException e) {
            // not available on this OS
        }
        return -1;
    }

    static double checksum(double[][] C) {
        double total = 0.0;
        for (double[] row : C) {
            double rs = 0.0;
            for (double v : row) rs += v;
            total += rs;
        }
        return total;
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 6) {
            System.err.println("Usage: MatMul A.bin B.bin n warmup reps batch [C_out.bin]");
            System.exit(2);
        }
        String aPath = args[0], bPath = args[1];
        int n = Integer.parseInt(args[2]);
        int warmup = Integer.parseInt(args[3]);
        int reps = Integer.parseInt(args[4]);
        int batch = Integer.parseInt(args[5]);
        String outPath = args.length > 6 ? args[6] : null;

        System.out.println("INFO,java,version," + System.getProperty("java.version"));
        System.out.println("INFO,java,vm," + System.getProperty("java.vm.name") + " " + System.getProperty("java.vm.version"));
        System.out.println("INFO,java,max_heap_bytes," + Runtime.getRuntime().maxMemory());

        double[][] A = readMatrix(aPath, n);
        double[][] B = readMatrix(bPath, n);
        double[][] C = new double[n][n]; // allocated outside the timer

        int total = warmup + reps;
        for (int r = 0; r < total; r++) {
            String phase = r < warmup ? "warmup" : "measured";
            int idx = r < warmup ? r : r - warmup;
            long t0 = System.nanoTime();
            for (int b = 0; b < batch; b++) {
                matmul(A, B, C, n);
            }
            long t1 = System.nanoTime();
            double cs = checksum(C); // consume the result (outside timer)
            System.out.printf("SAMPLE,java,%d,%s,%d,%d,%.1f,%s%n", n, phase, idx, batch,
                    (t1 - t0) / (double) batch, Double.toString(cs));
            System.out.flush();
        }
        Runtime rt = Runtime.getRuntime();
        System.out.println("INFO,java,heap_used_end_bytes," + (rt.totalMemory() - rt.freeMemory()));
        System.out.println("INFO,java,heap_committed_end_bytes," + rt.totalMemory());
        System.out.println("MEM,java," + n + ",vmhwm_kib," + peakRssKib());
        if (outPath != null) writeMatrix(outPath, C);
    }
}

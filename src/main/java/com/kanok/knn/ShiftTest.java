package com.kanok.knn;

import com.github.jelmerk.knn.DistanceFunctions;
import com.github.jelmerk.knn.SearchResult;
import com.github.jelmerk.knn.hnsw.HnswIndex;

import java.io.*;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class ShiftTest {

    public void recall() throws IOException {
        File file = new File("c:/All/VSB/2rocnik/AVD/sift1M/knnQA1M.bin");
        FileInputStream fin = new FileInputStream(file);
        BufferedInputStream bin = new BufferedInputStream(fin);
        DataInputStream din = new DataInputStream(bin);

        int count = (int) (file.length() / 4);
        int[] values = new int[count];
        for (int i = 0; i < count; i++) {
            System.out.println(din.readInt());
        }
        System.out.println(values[0]);
    }

    public void test() throws InterruptedException {
        //vectors count
        int nodeCount = 1000000;
        //query count
        int qSize = 10000;
        //vector dimension
        int vecDim = 128;
        //number of nearest neighbors to find
        int k = 10;

        HnswIndex<Integer, float[], Node, Float> hnswIndex = HnswIndex
                .newBuilder(vecDim, DistanceFunctions.FLOAT_EUCLIDEAN_DISTANCE, nodeCount)
                .withM(48)
                .withEf(500)
                .withEfConstruction(500)
                .build();

        /////////////////////////////////////////////////////// READ DATA
        FloatBuffer mass = createFloatBuffer("c:/All/VSB/2rocnik/AVD/sift1M/sift1M.bin");
        List<Node> nodes = createNodes(mass, nodeCount, vecDim);
        mass.clear();
        FloatBuffer massQ = createFloatBuffer("c:/All/VSB/2rocnik/AVD/sift1M/siftQ1M.bin");
        /////////////////////////////////////////////////////// QUERY PART

        System.out.println("Start building graph");
        long start = System.currentTimeMillis();
        hnswIndex.addAll(nodes, (workDone, max) -> System.out.printf("Added %d out of %d nodes to the index.%n", workDone, max));
        long end = System.currentTimeMillis();
        long buildTimeS = (end - start) / 1000;
        System.out.println("End building graph. Total build creating time: " + buildTimeS + " s");

        int count = 0;
        System.out.println("Start querying");
        start = System.currentTimeMillis();
        for (int i = 0; i < qSize; i++) {
            float[] massQArray = new float[128];
            massQ.get(massQArray, 0, vecDim);
            List<SearchResult<Node, Float>> nearest = hnswIndex.findNearest(massQArray, k);

            for (SearchResult<Node, Float> nodeFloatSearchResult : nearest) {
                System.out.println("Q number " + count + " AND ID: " + nodeFloatSearchResult.item().id());
            }
            count++;
        }
        end = System.currentTimeMillis();
        long totalMsQ = end - start;
        System.out.println("End querying. Q/s "+ qSize / (totalMsQ / 1000));
        System.out.println("Total ms " + totalMsQ);
        System.out.println();
        System.out.println("Total graph build time: " + buildTimeS + " s");
    }

    private List<Node> createNodes(FloatBuffer floatBuffer, int nodeCount, int vecDim) {
        List<Node> nodes = new ArrayList<>();
        int count = 0;
        for (int i = 0; i < nodeCount * vecDim; i += vecDim) {
            float[] floats = new float[vecDim];
            floatBuffer.get(floats, 0, vecDim);
            nodes.add(new Node(count++, floats));
        }
        return nodes;
    }

    private FloatBuffer createFloatBuffer(String filePath) {
        FloatBuffer mass = null;
        try (FileChannel fc = new RandomAccessFile(filePath, "rw").getChannel()) {
            mass = fc.map(FileChannel.MapMode.READ_WRITE, 0, fc.size())
                    .order(ByteOrder.nativeOrder()).asFloatBuffer();

        } catch (IOException e) {
            e.printStackTrace();
        }
        return mass;
    }


}

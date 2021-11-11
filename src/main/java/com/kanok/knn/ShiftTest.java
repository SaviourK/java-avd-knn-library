package com.kanok.knn;

import com.github.jelmerk.knn.DistanceFunctions;
import com.github.jelmerk.knn.SearchResult;
import com.github.jelmerk.knn.hnsw.HnswIndex;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class ShiftTest {

    public void test() throws InterruptedException, IOException {
        //vectors count
        int nodeCount = 1_000_000;
        //query count
        int qSize = 1_000;
        //vector dimension
        int vecDim = 128;
        //number of nearest neighbors to find
        int k = 10;

        HnswIndex<Integer, float[], Node, Float> hnswIndex = HnswIndex
                .newBuilder(vecDim, DistanceFunctions.FLOAT_EUCLIDEAN_DISTANCE, nodeCount)
                .withM(36)
                .withEf(150)
                .withEfConstruction(150)
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


        ObjectInputStream in = new ObjectInputStream(new FileInputStream("c:/All/VSB/2rocnik/AVD/sift1M/QA.bin"));

        System.out.println("Start querying");
        start = System.currentTimeMillis();
        float recall = 0;
        for (int i = 0; i < qSize; i++) {
            float[] massQArray = new float[128];
            massQ.get(massQArray, 0, vecDim);
            List<SearchResult<Node, Float>> nearest = hnswIndex.findNearest(massQArray, k);

            List<Integer> ni = new ArrayList<>();
            for (SearchResult<Node, Float> nodeFloatSearchResult : nearest) {
                ni.add(nodeFloatSearchResult.item().id());
            }

            List<Integer> nr = new ArrayList<>();
            for (int j = 0; j < 10; j++) {
                nr.add(in.readInt());
            }

            ni.retainAll(nr);
            recall += ni.size();

        }
        System.out.println("Recall: " + recall / (qSize * k));

        end = System.currentTimeMillis();
        long totalMsQ = end - start;
        System.out.println("Total ms " + totalMsQ);
        System.out.println("Total graph build time: " + buildTimeS + " s");

        in.close();
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

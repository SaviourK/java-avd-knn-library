package com.kanok.knn;

import com.github.jelmerk.knn.Item;

import java.util.Arrays;

public class Node implements Item<Integer, float[]> {

    private static final long serialVersionUID = 1L;

    private final Integer id;
    private final float[] vector;

    public Node(Integer id, float[] vector) {
        this.id = id;
        this.vector = vector;
    }

    @Override
    public Integer id() {
        return id;
    }

    @Override
    public float[] vector() {
        return vector;
    }

    @Override
    public int dimensions() {
        return vector.length;
    }

    @Override
    public String toString() {
        return "Node{" +
                "id='" + id + '\'' +
                ", vector=" + Arrays.toString(vector) +
                '}';
    }
}

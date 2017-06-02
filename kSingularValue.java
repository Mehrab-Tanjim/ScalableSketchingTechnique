/**
 * QCRI, sPCA LICENSE
 * sPCA is a scalable implementation of Principal Component Analysis (PCA) on of Spark and MapReduce
 *
 * Copyright (c) 2015, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
*/
package org.qcri.sparkpca;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.hadoop.io.IntWritable;
import org.apache.log4j.Level;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.QRDecomposition;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.storage.StorageLevel;
import org.qcri.sparkpca.FileFormat.OutputFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.Tuple2;

/**
 * This code provides an implementation of PPCA: Probabilistic Principal
 * Component Analysis based on the paper from Tipping and Bishop:
 * 
 * sPCA: PPCA on top of Spark
 * 
 * 
 * @author Tarek Elgamal
 * 
 */

public class kSingularValue implements Serializable {

	private final static Logger log = LoggerFactory.getLogger(kSingularValue.class);// getLogger(SparkPCA.class);
	
	static double k_plus_one_singular_value = 0;
	static int nClusters = 4;
	static int subsample = 10;
	static String dataset = "Untitled";
	static int q=2;//default

	public static void main(String[] args) throws IOException {
		org.apache.log4j.Logger.getLogger("org").setLevel(Level.ERROR);
		org.apache.log4j.Logger.getLogger("akka").setLevel(Level.ERROR);

		// Parsing input arguments
		final String inputPath;
		final String outputPath;
		final int nRows;
		final int nCols;
		final int nPCs;

		try {
			inputPath = System.getProperty("i");
			if (inputPath == null)
				throw new IllegalArgumentException();
		} catch (Exception e) {
			printLogMessage("i");
			return;
		}
		try {
			outputPath = System.getProperty("o");
			if (outputPath == null)
				throw new IllegalArgumentException();
		} catch (Exception e) {
			printLogMessage("o");
			return;
		}

		try {
			nRows = Integer.parseInt(System.getProperty("rows"));
		} catch (Exception e) {
			printLogMessage("rows");
			return;
		}

		
		try {
			nCols = Integer.parseInt(System.getProperty("cols"));
		} catch (Exception e) {
			printLogMessage("cols");
			return;
		}
		try {

			if (Integer.parseInt(System.getProperty("pcs")) == nCols) {
				nPCs = nCols - 1;
				System.out
						.println("Number of princpal components cannot be equal to number of dimension, reducing by 1");
			} else
				nPCs = Integer.parseInt(System.getProperty("pcs"));
		} catch (Exception e) {
			printLogMessage("pcs");
			return;
		}
		
		try {
			nClusters = Integer.parseInt(System.getProperty("clusters"));
			System.out.println("No of partition is set to" + nClusters);
		} catch (Exception e) {
			// log.warn("Cluster size is set to default: "+nClusters);
		}

		try {
			subsample = Integer.parseInt(System.getProperty("subsample"));
			System.out.println("No of subsample is set to" + subsample);
		} catch (Exception e) {
			// log.warn("Cluster size is set to default: "+nClusters);
		}

		
		try {
			q = Integer.parseInt(System.getProperty("q"));
			System.out.println("No of q is set to" + q);
		} catch (Exception e) {
			// log.warn("Cluster size is set to default: "+nClusters);
		}


		try {
			dataset = System.getProperty("dataset");
		} catch (IllegalArgumentException e) {
			// log.warn("Invalid Format " + System.getProperty("outFmt") + ",
			// Default name for dataset" + dataset + " will be used ");
		} catch (Exception e) {
			// log.warn("Default oname for dataset " + dataset + " will be used
			// ");
		}

		// Setting Spark configuration parameters
		SparkConf conf = new SparkConf().setAppName("kSingularValue").setMaster("local[*]");// TODO
																							// remove
																							// this
																							// part
																							// for
																							// building
		conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
		conf.set("spark.kryoserializer.buffer.max", "128m");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// compute principal components
		computeKSingularValue(sc, inputPath, outputPath, nRows, nCols, nPCs);

		// log.info("Principal components computed successfully ");
	}

	public static void computeKSingularValue(JavaSparkContext sc, String inputPath, String outputPath,final int nRows,
			final int nCols, final int nPCs) throws IOException {

		// Read from sequence file
		JavaPairRDD<IntWritable, VectorWritable> seqVectors = sc.sequenceFile(inputPath, IntWritable.class,
				VectorWritable.class, nClusters);

		JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors = seqVectors
				.map(new Function<Tuple2<IntWritable, VectorWritable>, org.apache.spark.mllib.linalg.Vector>() {

					public org.apache.spark.mllib.linalg.Vector call(Tuple2<IntWritable, VectorWritable> arg0)
							throws Exception {

						org.apache.mahout.math.Vector mahoutVector = arg0._2.get();
						Iterator<Element> elements = mahoutVector.nonZeroes().iterator();
						ArrayList<Tuple2<Integer, Double>> tupleList = new ArrayList<Tuple2<Integer, Double>>();
						while (elements.hasNext()) {
							Element e = elements.next();
							if (e.index() >= nCols || e.get() == 0)
								continue;
							Tuple2<Integer, Double> tuple = new Tuple2<Integer, Double>(e.index(), e.get());
							tupleList.add(tuple);
						}
						org.apache.spark.mllib.linalg.Vector sparkVector = Vectors.sparse(nCols, tupleList);
						return sparkVector;
					}
				}).persist(StorageLevel.MEMORY_ONLY_SER()); // TODO
															// change
															// later;

		// 1. Mean Job : This job calculates the mean and span of the columns of
		// the input RDD<org.apache.spark.mllib.linalg.Vector>
		final Accumulator<double[]> matrixAccumY = sc.accumulator(new double[nCols], new VectorAccumulatorParam());
		final double[] internalSumY = new double[nCols];
		vectors.foreachPartition(new VoidFunction<Iterator<org.apache.spark.mllib.linalg.Vector>>() {

			public void call(Iterator<org.apache.spark.mllib.linalg.Vector> arg0) throws Exception {
				org.apache.spark.mllib.linalg.Vector yi;
				int[] indices = null;
				int i;
				while (arg0.hasNext()) {
					yi = arg0.next();
					indices = ((SparseVector) yi).indices();
					for (i = 0; i < indices.length; i++) {
						internalSumY[indices[i]] += yi.apply(indices[i]);
					}
				}
				matrixAccumY.add(internalSumY);
			}

		});// End Mean Job

		// Get the sum of column Vector from the accumulator and divide each
		// element by the number of rows to get the mean
		// not best of practice to use non-final variable
		final Vector meanVector = new DenseVector(matrixAccumY.value()).divide(nRows);
		final Broadcast<Vector> br_ym_mahout = sc.broadcast(meanVector);

		JavaRDD<org.apache.spark.mllib.linalg.Vector> A = vectors
				.map(new Function<org.apache.spark.mllib.linalg.Vector, org.apache.spark.mllib.linalg.Vector>() {

					public org.apache.spark.mllib.linalg.Vector call(org.apache.spark.mllib.linalg.Vector arg0)
							throws Exception {

						double[] ans = new double[arg0.size()];

						for (int i = 0; i < arg0.size(); i++) {
							ans[i] = arg0.apply(i) - br_ym_mahout.value().getQuick(i);
						}
						org.apache.spark.mllib.linalg.Vector sparkVector = org.apache.spark.mllib.linalg.Vectors
								.dense(ans);
						return sparkVector;
					}
				});

		k_plus_one_singular_value = new Norm().spectralNorm(sc, A, nRows, nCols, nPCs+1, subsample, q, outputPath, dataset);
		
		String fileLocation = outputPath+ File.separator +"kSingularValue.txt"; 
		File f = new File(fileLocation);
		PrintWriter out = null;
		if ( f.exists() && !f.isDirectory() ) {
		    out = new PrintWriter(new FileOutputStream(f, true));
		    out.println("kth Singular Value for "+dataset+" is :"+k_plus_one_singular_value);
		}
		else{
			out = new PrintWriter(fileLocation);
			out.append("kth Singular Value for "+dataset+" is :"+k_plus_one_singular_value);
		}
		out.close();
	}

	private static void printLogMessage(String argName) {
		log.error("Missing arguments -D" + argName);
		log.info("Usage: -Di=<path/to/input/matrix> -Do=<path/to/outputfolder> -Drows=<number of rows> -Dcols=<number of columns> -Dpcs=<number of principal components> [-DerrSampleRate=<Error sampling rate>] [-DmaxIter=<max iterations>] [-DoutFmt=<output format>] [-DComputeProjectedMatrix=<0/1 (compute projected matrix or not)>]");
	}
}
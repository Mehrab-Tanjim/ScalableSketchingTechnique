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
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
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

import com.esotericsoftware.minlog.Log;

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

public class Norm implements Serializable {

	public static double spectralNorm(JavaSparkContext sc, final JavaRDD<org.apache.spark.mllib.linalg.Vector> A,
			final int nRows, final int nCols, final int nPCs, final int subsample, final int q, 
			final String outputPath, final String appname) throws FileNotFoundException {

		String fileLocation = outputPath + File.separator + "kSingularValueLog.txt";
		File f = new File(fileLocation);

		PrintWriter out = null;
		if (f.exists() && !f.isDirectory()) {
			out = new PrintWriter(new FileOutputStream(f, true));
			out.append("\n" + appname+"\n");
		} else {
			out = new PrintWriter(fileLocation);
			out.println(appname);
		}

		/************************** SSVD PART *****************************/

		// initialize & broadcast a random seed
		org.apache.spark.mllib.linalg.Matrix GaussianRandomMatrix = org.apache.spark.mllib.linalg.Matrices.randn(nCols,
				nPCs + subsample, new java.util.Random(System.currentTimeMillis()));
		Broadcast<org.apache.spark.mllib.linalg.Matrix> br_QtA = sc.broadcast(GaussianRandomMatrix);

		// derive the sketched matrix, B=A*S
		final RowMatrix matA = new RowMatrix(A.rdd());

		RowMatrix y = null;
		QRDecomposition<RowMatrix, org.apache.spark.mllib.linalg.Matrix> qr = null;
		JavaPairRDD<org.apache.spark.mllib.linalg.Vector, org.apache.spark.mllib.linalg.Vector> QA = null;
		JavaRDD<org.apache.spark.mllib.linalg.Matrix> QTA_partial = null;
		org.apache.mahout.math.SingularValueDecomposition svd = null;
		for (int iter = 0; iter < q; iter++) {
			y = matA.multiply((org.apache.spark.mllib.linalg.Matrix) br_QtA.getValue());
			// QR decomposition of B
			qr = y.tallSkinnyQR(true);

			QA = qr.Q().rows().toJavaRDD().zip(A);
			
			final Accumulator<double[]> sumQtA = sc.accumulator(new double[(nPCs + subsample) * nCols],
					new VectorAccumulatorParam());

			final double[] sumQtAPartial = new double[(nPCs + subsample) * nCols];

			QA.foreachPartition(
					new VoidFunction<Iterator<Tuple2<org.apache.spark.mllib.linalg.Vector, org.apache.spark.mllib.linalg.Vector>>>() {

						@Override
						public void call(
								Iterator<Tuple2<org.apache.spark.mllib.linalg.Vector, org.apache.spark.mllib.linalg.Vector>> arg0)
								throws Exception {

							Tuple2<org.apache.spark.mllib.linalg.Vector, org.apache.spark.mllib.linalg.Vector> pair;
							double[] A = null;
							double[] Q = null;
							while (arg0.hasNext()) {
								// lol mistake
								pair = arg0.next();
								A = pair._2.toArray();// TODO check does it really
														// save time, and why?!?!?!
								Q = pair._1.toArray();

								int row = Q.length;
								int column = A.length;
								for (int j = 0; j < column; j++) {
									for (int i = 0; i < row; i++) {
										sumQtAPartial[row * j + i] += Q[i] * A[j];
									}
								}

							}

							sumQtA.add(sumQtAPartial);

						}

					});
			
			double[][] QtA = new double[nPCs + subsample][nCols];
			
			for (int i = 0; i < (nPCs + subsample); i++) {
				for (int j = 0; j < nCols; j++) {
					QtA[i][j] = sumQtA.value()[(nPCs + subsample) * j + i];
				}
			}
			Matrix B = new DenseMatrix(QtA);
			out.append("mapping part done");
			out.flush();
			
			br_QtA = sc.broadcast(PCAUtils.convertMahoutToSparkMatrix(B).transpose());
			svd = new org.apache.mahout.math.SingularValueDecomposition(B);
			// keeping log
			out.append("Specral Norm of " + (nPCs - 1) + " after " + iter + " iteration is :"
					+ svd.getS().getQuick(nPCs - 1, nPCs - 1)+"\n");
			out.flush();//lololol

		}

		out.append("************");
		out.close();
		return svd.getS().getQuick(nPCs - 1, nPCs - 1);
	}
	/*
	 * // // /** // * Sketch dimension ,S=nPCs+subsample Sketched matrix, B=A*S;
	 * QR // * decomposition, Q=qr(B); SV decomposition, [~,s,V]=svd(Q); //
	 */
	//
	// // initialize & broadcast a random seed
	// org.apache.spark.mllib.linalg.Matrix GaussianRandomMatrix =
	// org.apache.spark.mllib.linalg.Matrices.randn(nCols,
	// nPCs + subsample, new SecureRandom());
	// //PCAUtils.printMatrixToFile(GaussianRandomMatrix, OutputFormat.DENSE,
	// outputPath+File.separator+"Seed");
	// final Matrix seedMahoutMatrix =
	// PCAUtils.convertSparkToMahoutMatrix(GaussianRandomMatrix);
	// final Broadcast<Matrix> seed = sc.broadcast(seedMahoutMatrix);
	//
	// JavaRDD<org.apache.spark.mllib.linalg.Vector> Y = vectors
	// .map(new Function<org.apache.spark.mllib.linalg.Vector,
	// org.apache.spark.mllib.linalg.Vector>() {
	//
	// public org.apache.spark.mllib.linalg.Vector
	// call(org.apache.spark.mllib.linalg.Vector arg0)
	// throws Exception {
	// double[] y = new double[nPCs + subsample];
	// double[] values = arg0.toArray();// TODO check does it
	// // really save
	// // time?!?!
	//
	// int[] indices = ((SparseVector) arg0).indices();
	// int index;
	// double value = 0;
	// for (int j = 0; j < (nPCs + subsample); j++) {
	// for (int i = 0; i < indices.length; i++) {
	// index = indices[i];
	// value += values[index] * seed.value().getQuick(index, j);
	// }
	// y[j] = value - brSeedMu.value().getQuick(j);
	// value = 0;
	// }
	//
	// return Vectors.dense(y);
	//
	// }
	// });
	//
	// // QR decomposition of B
	// QRDecomposition<RowMatrix, org.apache.spark.mllib.linalg.Matrix> QR = new
	// RowMatrix(Y.rdd()).tallSkinnyQR(true);
	//
	// JavaPairRDD<org.apache.spark.mllib.linalg.Vector,
	// org.apache.spark.mllib.linalg.Vector> QnA = QR.Q().rows()
	// .toJavaRDD().zip(vectors);
	// final Accumulator<double[]> sumQ = sc.accumulator(new double[nPCs +
	// subsample], new VectorAccumulatorParam());
	// final Accumulator<double[]> sumQtA = sc.accumulator(new double[(nPCs +
	// subsample) * nCols],
	// new VectorAccumulatorParam());
	//
	// final double[] sumQPartial = new double[nPCs + subsample];
	// final double[] sumQtAPartial = new double[(nPCs + subsample) * nCols];
	//
	// QnA.foreachPartition(
	// new VoidFunction<Iterator<Tuple2<org.apache.spark.mllib.linalg.Vector,
	// org.apache.spark.mllib.linalg.Vector>>>() {
	//
	// @Override
	// public void call(
	// Iterator<Tuple2<org.apache.spark.mllib.linalg.Vector,
	// org.apache.spark.mllib.linalg.Vector>> arg0)
	// throws Exception {
	//
	// Tuple2<org.apache.spark.mllib.linalg.Vector,
	// org.apache.spark.mllib.linalg.Vector> pair;
	// double[] A = null;
	// double[] Q = null;
	// while (arg0.hasNext()) {
	// // lol mistake
	// pair = arg0.next();
	// A = pair._2.toArray();// TODO check does it really
	// // save time, and why?!?!?!
	// Q = pair._1.toArray();
	//
	// int row = Q.length;
	// int[] indices = ((SparseVector) pair._2).indices();
	// int index;
	// for (int j = 0; j < indices.length; j++) {
	// for (int i = 0; i < row; i++) {
	// index = indices[j];
	// sumQtAPartial[row * index + i] += Q[i] * A[index];
	// }
	// }
	// for (int i = 0; i < row; i++) {
	// sumQPartial[i] += Q[i];
	// }
	//
	// }
	//
	// sumQ.add(sumQPartial);
	// sumQtA.add(sumQtAPartial);
	//
	// }
	//
	// });
	//
	// double[][] QtA = new double[nPCs + subsample][nCols];
	//
	//
	// ///iteration
	//
	//
	//
	// for (int i = 0; i < (nPCs + subsample); i++) {
	// for (int j = 0; j < nCols; j++) {
	// QtA[i][j] = sumQtA.value()[(nPCs + subsample) * j + i]
	// - sumQ.value()[i] * br_ym_mahout.value().getQuick(j);
	// }
	// }
	// Matrix B = new DenseMatrix(QtA);
	// org.apache.mahout.math.SingularValueDecomposition SVD = new
	// org.apache.mahout.math.SingularValueDecomposition(
	// B);
	//
	// /* clean up */
	// seed.destroy();
	// return SVD.getS().getQuick(nPCs-1, nPCs-1);

}
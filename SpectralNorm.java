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

public class SpectralNorm implements Serializable {

	
	public static double norm(JavaSparkContext sc, JavaRDD<org.apache.spark.mllib.linalg.Vector> vectors,
			final int nRows, final int nCols, final int nPCs, final int subsample, final int maxIterations) {

		
		/************************** SSVD PART *****************************/

		/**
		 * Sketch dimension ,S=nPCs+subsample Sketched matrix, B=A*S; QR
		 * decomposition, Q=qr(B); SV decomposition, [~,s,V]=svd(Q);
		 */

		// initialize & broadcast a random seed
		org.apache.spark.mllib.linalg.Matrix GaussianRandomMatrix = org.apache.spark.mllib.linalg.Matrices.randn(nCols,
				nPCs + subsample, new SecureRandom());
		//PCAUtils.printMatrixToFile(GaussianRandomMatrix, OutputFormat.DENSE, outputPath+File.separator+"Seed");
		final Matrix seedMahoutMatrix = PCAUtils.convertSparkToMahoutMatrix(GaussianRandomMatrix);
		final Broadcast<Matrix> seed = sc.broadcast(seedMahoutMatrix);
		
		JavaRDD<org.apache.spark.mllib.linalg.Vector> Y = vectors
				.map(new Function<org.apache.spark.mllib.linalg.Vector, org.apache.spark.mllib.linalg.Vector>() {

					public org.apache.spark.mllib.linalg.Vector call(org.apache.spark.mllib.linalg.Vector arg0)
							throws Exception {
						double[] y = new double[nPCs + subsample];
						double[] values = arg0.toArray();// TODO check does it
															// really save
															// time?!?!

						int[] indices = ((SparseVector) arg0).indices();
						int index;
						double value = 0;
						for (int j = 0; j < (nPCs + subsample); j++) {
							for (int i = 0; i < indices.length; i++) {
								index = indices[i];
								value += values[index] * seed.value().getQuick(index, j);
							}
							y[j] = value - brSeedMu.value().getQuick(j);
							value = 0;
						}

						return Vectors.dense(y);

					}
				});

		// QR decomposition of B
		QRDecomposition<RowMatrix, org.apache.spark.mllib.linalg.Matrix> QR = new RowMatrix(Y.rdd()).tallSkinnyQR(true);

		JavaPairRDD<org.apache.spark.mllib.linalg.Vector, org.apache.spark.mllib.linalg.Vector> QnA = QR.Q().rows()
				.toJavaRDD().zip(vectors);
		final Accumulator<double[]> sumQ = sc.accumulator(new double[nPCs + subsample], new VectorAccumulatorParam());
		final Accumulator<double[]> sumQtA = sc.accumulator(new double[(nPCs + subsample) * nCols],
				new VectorAccumulatorParam());

		final double[] sumQPartial = new double[nPCs + subsample];
		final double[] sumQtAPartial = new double[(nPCs + subsample) * nCols];

		QnA.foreachPartition(
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
							int[] indices = ((SparseVector) pair._2).indices();
							int index;
							for (int j = 0; j < indices.length; j++) {
								for (int i = 0; i < row; i++) {
									index = indices[j];
									sumQtAPartial[row * index + i] += Q[i] * A[index];
								}
							}
							for (int i = 0; i < row; i++) {
								sumQPartial[i] += Q[i];
							}

						}

						sumQ.add(sumQPartial);
						sumQtA.add(sumQtAPartial);

					}

				});

		double[][] QtA = new double[nPCs + subsample][nCols];
		
		
		///iteration
		
		

		for (int i = 0; i < (nPCs + subsample); i++) {
			for (int j = 0; j < nCols; j++) {
				QtA[i][j] = sumQtA.value()[(nPCs + subsample) * j + i]
						- sumQ.value()[i] * br_ym_mahout.value().getQuick(j);
			}
		}
		Matrix B = new DenseMatrix(QtA);
		org.apache.mahout.math.SingularValueDecomposition SVD = new org.apache.mahout.math.SingularValueDecomposition(
				B);		

		/* clean up */
		seed.destroy();
		return SVD.getS().getQuick(nPCs-1, nPCs-1);
	}

	
}
package r1p.test.myapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private CascadeClassifier classifier;
    private Mat mRgba;
    private Mat img;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        CameraBridgeViewBase cameraView = findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this);
        cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
        cameraView.enableView();

         if (OpenCVLoader.initDebug()) {
             initClassifier();
        }
    }
    private void initClassifier() {
        try {
            InputStream is = getResources()
                    .openRawResource(R.raw.haarcascade_frontalface_alt);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            classifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        try {
            img = Utils.loadResource(this, R.drawable.glasses);
            Imgproc.cvtColor(img, img, Imgproc.COLOR_RGBA2BGRA, 4);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        int mAbsoluteFaceSize = 0;
        float mRelativeFaceSize = 0.2f;
        int height = inputFrame.gray().rows();
        if (Math.round(height * mRelativeFaceSize) > 0) {
            mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
        }
        MatOfRect faces = new MatOfRect();
        if (classifier != null)
            classifier.detectMultiScale(inputFrame.gray(), faces, 1.1, 2, 2,
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        Rect[] facesArray = faces.toArray();
        Scalar faceRectColor = new Scalar(0, 255, 0, 255); // color for rectangle
        for (Rect faceRect : facesArray){
            Rect rectCrop = new Rect(faceRect.x, faceRect.y, faceRect.width, faceRect.height);
            Mat resizedImg = new Mat();
            Size imgNewSize = new Size(faceRect.width, faceRect.height);
            Imgproc.resize(img, resizedImg, imgNewSize, 0, 0, Imgproc.INTER_AREA);
            Mat threshold = new Mat();
            Imgproc.threshold(resizedImg, threshold , 240, 255, Imgproc.THRESH_BINARY_INV);
//            Imgproc.rectangle(mRgba, faceRect.tl(), faceRect.br(), faceRectColor, 3); // rectangle around the face
            Mat imageROI = mRgba.submat(rectCrop);
            resizedImg.copyTo(imageROI, threshold);
        }
        return mRgba;
    }
}

package solutionanalysis.util.views;

import android.media.ImageReader;

public abstract class ImageCaptureListener implements Runnable {

    ImageReader reader;

    @Override
    public void run() {

        onCaptured(reader);
    }

    protected abstract void onCaptured(ImageReader reader);
}

package solutionanalysis.display_app;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.media.ImageReader;
import android.util.Log;
import android.widget.ImageView;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;

import solutionanalysis.util.EasyToast;
import solutionanalysis.util.Initializer;
import solutionanalysis.util.Permission;
import solutionanalysis.util.networking.ConnectionManager;
import solutionanalysis.util.networking.ERequest;
import solutionanalysis.util.networking.EResponse;
import solutionanalysis.util.networking.IConnectionErrorListener;
import solutionanalysis.util.networking.IInteractionHandler;
import solutionanalysis.util.networking.RequestBundle;
import solutionanalysis.util.views.CameraPreview;
import solutionanalysis.util.views.ImageCaptureListener;

public class MainActivity extends AppCompatActivity implements IInteractionHandler, IConnectionErrorListener {

    private static final String TAG = "MainActivity";

    private ImageView imageView;
    private CameraPreview preview;

    @Override
    protected void onCreate(android.os.Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Initializer.init(getApplicationContext());
        Permission.requestPermissionsIfNotGranted(this);

        imageView = findViewById(R.id.main_image_view);
        preview = findViewById(R.id.main_preview);
    }

    @Override
    protected void onStart() {
        super.onStart();

        ConnectionManager manager = ConnectionManager.getInstance();
        if (manager != null) {
            manager.setInteractionHandler(this);
            manager.setConnectionErrorListener(this);
            manager.connect(ERequest.DISPLAY);
        } else {
            Log.e(TAG, "ConnectionManager service not running");
        }

        preview.setCameraId("1").onResume(this);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        ConnectionManager.getInstance().request(ERequest.ANY_QUIT, new byte[0]);
    }

    @Override
    public void onAlreadyConnected() {
        Log.d(TAG, "onAlreadyConnected");
    }

    @Override
    public void onHostUnreachable() {
        Log.d(TAG, "onHostUnreachable");
    }

    @Override
    public void handleRequest(final ConnectionManager conn, RequestBundle bundle) {

        switch (bundle.request) {
            case DISPLAY_TAKE_PICTURE:
                preview.takePicture(new ImageCaptureListener() {
                    @Override
                    protected void onCaptured(ImageReader reader) {

                        ByteBuffer buffer = reader.acquireLatestImage().getPlanes()[0].getBuffer();
                        byte[] imageBytes = new byte[buffer.capacity()];
                        Log.d(TAG, "Image Size: " + imageBytes.length + " bytes");
                        buffer.get(imageBytes);
                        bundle.response = EResponse.OK;
                        bundle.args = imageBytes;
                        conn.response(bundle);
                    }
                });
                return;

            case DISPLAY_SHOW_PICTURE:
                ByteArrayInputStream imageStream = new ByteArrayInputStream(bundle.args);
                Bitmap bmp = BitmapFactory.decodeStream(imageStream);

                if (bmp != null) {{
                    bundle.response = EResponse.ERROR;
                }
                    getMainExecutor().execute(() -> {
                        imageView.setImageBitmap(bmp);
                    });
                    bundle.response = EResponse.OK;
                } else

                bundle.args = new byte[0];
                conn.response(bundle);
                break;

            case ANY_QUIT:
                bundle.response = EResponse.ACK;
                conn.response(bundle);
                EasyToast.showLong("Application terminated due to a request from the host.");
                finishAndRemoveTask();
                break;

            default:
                break;
        }
    }

    @Override
    public void handleResponse(RequestBundle bundle) {

        if (bundle.request == ERequest.ANY_QUIT) {
            // When this client requested quit.
            finishAndRemoveTask();
        } else {
            Log.e(TAG, "Unable request code.");
        }
    }
}
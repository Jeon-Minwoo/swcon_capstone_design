package solutionanalysis.camera_app;

import androidx.appcompat.app.AppCompatActivity;

import android.media.ImageReader;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.GridLayout;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;

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
import solutionanalysis.util.views.PreviewFragment;

public class MainActivity extends AppCompatActivity implements IInteractionHandler, IConnectionErrorListener {

    private static final String TAG = "MainActivity";

    private GridLayout container;
    private ArrayList<CameraPreview> previews = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Initializer.init(getApplicationContext());
        Permission.requestPermissionsIfNotGranted(this);

        container = findViewById(R.id.main_preview_container);
    }

    @Override
    protected void onStart() {
        super.onStart();

        ConnectionManager manager = ConnectionManager.getInstance();
        if (manager != null) {
            manager.setInteractionHandler(this);
            manager.setConnectionErrorListener(this);
            manager.connect(ERequest.CAMERA);
        } else {
            Log.e(TAG, "ConnectionManager service not running");
        }

        addPreview("0");
        addPreview("1");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        ConnectionManager.getInstance().request(ERequest.ANY_QUIT, new byte[0]);
    }

    private void addPreview(String cameraId) {

        View view = findViewById(R.id.main_root);
        view.post(() -> {
            final int FragmentSize = Math.min(view.getWidth(), view.getHeight()) / 2;
            final PreviewFragment fragment = PreviewFragment.newInstance(cameraId, FragmentSize, FragmentSize);
            fragment.onViewCreated = () -> {
                previews.add(fragment.getPreview());
            };

            getSupportFragmentManager().beginTransaction()
                    .add(R.id.main_preview_container, fragment)
                    .commit();
        });
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

        Log.d(TAG, "GotRequest: " + bundle.toString());
        switch (bundle.request) {
            case CAMERA_TAKE_PICTURE:
                final int cameraIdx = bundle.args[0];
                if (cameraIdx < 0 || previews.size() <= cameraIdx) {
                    Log.e(TAG, "Camera index out of range.");
                    bundle.response = EResponse.ERROR;
                } else {
                    previews.get(cameraIdx).takePicture(new ImageCaptureListener() {
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
                }
                break;

            case CAMERA_TOGGLE_TORCH:
                previews.get(0).toggleTorch();
                bundle.response = EResponse.OK;
                conn.response(bundle);
                break;

            case ANY_QUIT:
                bundle.response = EResponse.ACK;
                conn.response(bundle);
                EasyToast.showLong("Application terminated due to a request from the host.");
                finishAndRemoveTask();
                break;

            default:
                bundle.response = EResponse.REJECT;
                conn.response(bundle);
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
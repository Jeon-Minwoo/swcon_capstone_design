package solutionanalysis.util.views;

import android.annotation.SuppressLint;
import android.os.Bundle;

import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import solutionanalysis.util.R;

public class PreviewFragment extends Fragment {

    private static final String PARAM_CAMERA_ID = "camera_id";
    private static final String PARAM_VIEW_WIDTH = "view_width";
    private static final String PARAM_VIEW_HEIGHT = "view_height";

    private String cameraId;
    private int width, height;

    private CameraPreview previewView;
    private TextView idView;

    public Runnable onViewCreated = null;

    public PreviewFragment() {
        // Required empty public constructor
    }

    public static PreviewFragment newInstance(String cameraId, int width, int height) {
        PreviewFragment fragment = new PreviewFragment();
        fragment.cameraId = cameraId;
        fragment.width = width;
        fragment.height = height;

        Bundle args = new Bundle();
        args.putString(PARAM_CAMERA_ID, cameraId);
        args.putInt(PARAM_VIEW_WIDTH, width);
        args.putInt(PARAM_VIEW_HEIGHT, height);

        fragment.setArguments(args);

        return fragment;
    }

    @SuppressLint("SetTextI18n")
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View view = inflater.inflate(R.layout.fragment_preview, container, false);

        ViewGroup.LayoutParams layoutParams = view.getLayoutParams();
        layoutParams.width = width;
        layoutParams.height = height;
        view.setLayoutParams(layoutParams);

        previewView = view.findViewById(R.id.preview_preview_view);
        previewView.setCameraId(cameraId);
        previewView.onResume(requireActivity());

        idView = view.findViewById(R.id.preview_id_view);
        idView.setText("Camera ID: " + cameraId);

        if (onViewCreated != null) {
            onViewCreated.run();
        }

        return view;
    }

    public CameraPreview getPreview() {

        return previewView;
    }
}
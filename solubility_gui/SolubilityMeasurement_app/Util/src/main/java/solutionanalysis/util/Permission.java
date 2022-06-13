package solutionanalysis.util;

import android.content.Context;
import android.content.pm.PackageManager;

import androidx.appcompat.app.AppCompatActivity;

import java.util.Arrays;
import java.util.Optional;

public final class Permission {

    private static Context AppContext = null;

    private static final String[] REQUIRED_PERMISSIONS = {
            "android.permission.CAMERA",
            "android.permission.WRITE_EXTERNAL_STORAGE",
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.INTERNET",
            "android.permission.ACCESS_NETWORK_STATE" };
    public static final int PERMISSION_REQUEST_CODE = 1001;

    static void initContext(Context context) {
        AppContext = context;
    }

    public static boolean checkPermissions() {
        Optional<Boolean> bGranted = Arrays.stream(REQUIRED_PERMISSIONS)
                .map((String str) -> AppContext.checkSelfPermission(str) == PackageManager.PERMISSION_GRANTED)
                .reduce(Boolean::logicalAnd);

        return bGranted.orElse(false);
    }

    public static void requestPermissions(AppCompatActivity activity) {

        String[] requiredPermissions = (String[]) Arrays.stream(REQUIRED_PERMISSIONS)
                .filter(perm -> activity.checkSelfPermission(perm) != PackageManager.PERMISSION_GRANTED)
                .toArray(String[]::new);

        activity.requestPermissions(requiredPermissions, PERMISSION_REQUEST_CODE);
    }

    public static void requestPermissionsIfNotGranted(AppCompatActivity activity) {

        if (!checkPermissions()) {
            requestPermissions(activity);
        }
    }
}

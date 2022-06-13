package solutionanalysis.util;

import android.content.Context;
import android.widget.Toast;

public final class EasyToast {

    private static Context AppContext = null;

    static void initContext(Context context) {
        AppContext = context;
    }

    public static void showLong(String message) {
        Toast.makeText(AppContext, message, Toast.LENGTH_LONG).show();
    }

    public static void showShort(String message) {
        Toast.makeText(AppContext, message, Toast.LENGTH_SHORT).show();
    }
}

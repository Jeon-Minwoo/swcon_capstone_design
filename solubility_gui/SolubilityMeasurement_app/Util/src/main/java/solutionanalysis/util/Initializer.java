package solutionanalysis.util;

import android.content.Context;

import androidx.annotation.NonNull;

import solutionanalysis.util.networking.ConnectionManager;

public final class Initializer {

    public static void init(@NonNull Context context) {
        EasyToast.initContext(context);
        Permission.initContext(context);
    }
}

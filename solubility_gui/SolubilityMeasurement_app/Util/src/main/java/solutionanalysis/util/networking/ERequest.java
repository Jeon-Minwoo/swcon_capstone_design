package solutionanalysis.util.networking;

public enum ERequest {

    NONE(0x00),
    CAMERA(0x01), DISPLAY(0x02),
    ANY(CAMERA.value | DISPLAY.value),

    CAMERA_TAKE_PICTURE(CAMERA.value | 0x10),
    CAMERA_TOGGLE_TORCH(CAMERA.value | 0x20),

    DISPLAY_TAKE_PICTURE(DISPLAY.value | 0x10),
    DISPLAY_SHOW_PICTURE(DISPLAY.value | 0x20),

    ANY_QUIT(ANY.value | 0x10);

    public static ERequest fromByte(byte data) {

        for (ERequest req : ERequest.values()) {
            if (req.value == data) {
                return req;
            }
        }

        return null;
    }

    private final byte value;

    ERequest(int value) {

        assert Byte.MIN_VALUE <= value && value <= Byte.MAX_VALUE;

        this.value = (byte)value;
    }

    public byte getValue() {

        return this.value;
    }
}

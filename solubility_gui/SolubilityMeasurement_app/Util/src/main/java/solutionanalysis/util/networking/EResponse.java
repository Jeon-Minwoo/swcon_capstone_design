package solutionanalysis.util.networking;

public enum EResponse {

    NONE(0),
    OK(1), ACK(2),
    REJECT(3), ERROR(4);

    public static EResponse fromByte(byte data) {

        for (EResponse req : EResponse.values()) {
            if (req.value == data) {
                return req;
            }
        }

        return null;
    }

    private final byte value;

    EResponse(int value) {

        assert Byte.MIN_VALUE <= value && value <= Byte.MAX_VALUE;

        this.value = (byte)value;
    }

    public byte getValue() {

        return this.value;
    }
}

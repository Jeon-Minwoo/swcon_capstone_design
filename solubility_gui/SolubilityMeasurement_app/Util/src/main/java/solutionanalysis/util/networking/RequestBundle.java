package solutionanalysis.util.networking;

import androidx.annotation.NonNull;

import java.util.Arrays;

public final class RequestBundle {

    public static RequestBundle fromBytes(byte[] data) {

        byte requestId = data[0];
        ERequest request = ERequest.fromByte(data[1]);
        EResponse response = EResponse.fromByte(data[2]);
        byte[] args = Arrays.copyOfRange(data, 3, data.length);

        return new RequestBundle(requestId, request, args, response);
    }

    public byte requestId;
    public ERequest request;
    public byte[] args;
    public EResponse response;

    public RequestBundle(byte requestId, ERequest request, byte[] args, EResponse response) {

        this.requestId = requestId;
        this.request = request;
        this.args = args;
        this.response = response;
    }

    public byte[] toBytes() {

        byte[] data = new byte[3 + args.length];
        data[0] = requestId;
        data[1] = request.getValue();
        data[2] = response.getValue();
        System.arraycopy(args, 0, data, 3, args.length);

        return data;
    }

    @NonNull
    @Override
    public String toString() {
        return "RequestBundle{" +
                "requestId=" + requestId +
                ", request=" + request.getValue() +
                ", args=" + (args == null || args.length == 0 ? "Empty" : "Byte[]") +
                ", response=" + response.getValue() +
                '}';
    }
}

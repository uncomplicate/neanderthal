package uncomplicate.neanderthal.protocols;

public interface RealBufferAccessor {

    double get (ByteBuffer buf, long index);

    void set (ByteBuffer buf, long index, double value);

    long elementBytes ();
}

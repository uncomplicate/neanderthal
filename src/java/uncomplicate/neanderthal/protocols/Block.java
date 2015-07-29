package uncomplicate.neanderthal.protocols;

public interface Block {

    Object buffer ();

    BLAS engine ();

    long elementBytes ();

    Object elementType ();

    long stride ();

    long order ();

    boolean compatible (Block y);

}

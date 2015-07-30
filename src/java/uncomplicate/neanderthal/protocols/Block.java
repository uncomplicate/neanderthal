package uncomplicate.neanderthal.protocols;

public interface Block {

    Object buffer ();

    BLAS engine ();

    long elementBytes ();
    // TODO one of these 2 may not be needed
    Object elementType ();

    long stride ();

    long order ();

    boolean compatible (Block y);

}

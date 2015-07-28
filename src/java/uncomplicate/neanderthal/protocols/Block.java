package uncomplicate.neanderthal.protocols;

public interface Block {

    Object buffer ();

    long elementBytes ();

    long stride ();

    boolean compatible (Block y);

    long order ();
}

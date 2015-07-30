package uncomplicate.neanderthal.protocols;

public interface Block {

    Object buffer ();

    Object elementType ();

    long stride ();

    long order ();

    long count ();

}

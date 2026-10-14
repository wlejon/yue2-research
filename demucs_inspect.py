"""Print htdemucs_ft's bag: per-model kwargs and state-dict shapes (porting aid).

    .venv/Scripts/python demucs_inspect.py [--shapes]
"""
import sys

from demucs.pretrained import get_model


def main():
    bag = get_model("htdemucs_ft")
    print("sources", bag.sources, "sr", bag.samplerate, "weights", bag.weights)
    for i, m in enumerate(bag.models):
        print(f"--- model {i}: {type(m).__name__} segment={m.segment} sources={m.sources}")
        kw = m._init_args_kwargs if hasattr(m, "_init_args_kwargs") else None
        print("kwargs", kw)
        if "--shapes" in sys.argv and i == 0:
            for k, v in m.state_dict().items():
                print(k, tuple(v.shape))
        print("n params", sum(p.numel() for p in m.parameters()))


if __name__ == "__main__":
    main()

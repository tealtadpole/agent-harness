import { useEffect, useRef, useState } from "react";

export default function Composer({
  disabled,
  placeholder,
  onSend,
}: {
  disabled: boolean;
  placeholder: string;
  onSend: (text: string) => void;
}) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [text]);

  useEffect(() => {
    if (!disabled) ref.current?.focus();
  }, [disabled]);

  const submit = () => {
    const value = text.trim();
    if (!value || disabled) return;
    onSend(value);
    setText("");
  };

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <textarea
        ref={ref}
        rows={1}
        value={text}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
      />
      <button className="btn-primary" type="submit" disabled={disabled || !text.trim()}>
        Send
      </button>
    </form>
  );
}

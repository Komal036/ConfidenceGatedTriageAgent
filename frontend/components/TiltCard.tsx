"use client";

import { useRef, useState } from "react";

const MAX_TILT = 6; // degrees, kept subtle deliberately

export default function TiltCard({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [style, setStyle] = useState<React.CSSProperties>({});
  const [glare, setGlare] = useState({ x: 50, y: 50, opacity: 0 });

  function handleMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width; // 0..1
    const py = (e.clientY - rect.top) / rect.height; // 0..1

    const rotateY = (px - 0.5) * MAX_TILT * 2;
    const rotateX = (0.5 - py) * MAX_TILT * 2;

    setStyle({
      transform: `perspective(900px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(0)`,
      transition: "transform 60ms ease-out",
    });
    setGlare({ x: px * 100, y: py * 100, opacity: 1 });
  }

  function handleMouseLeave() {
    setStyle({
      transform: "perspective(900px) rotateX(0deg) rotateY(0deg) translateZ(0)",
      transition: "transform 400ms cubic-bezier(0.22, 1, 0.36, 1)",
    });
    setGlare((g) => ({ ...g, opacity: 0 }));
  }

  return (
    <div
      ref={ref}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{ ...style, transformStyle: "preserve-3d" }}
      className={`relative will-change-transform ${className}`}
    >
      {children}
      {/* Cursor-tracked glare, sits above content, ignores pointer events */}
      <div
        className="pointer-events-none absolute inset-0 rounded-[inherit] transition-opacity duration-300"
        style={{
          opacity: glare.opacity * 0.5,
          background: `radial-gradient(circle at ${glare.x}% ${glare.y}%, rgba(232,236,239,0.08), transparent 55%)`,
        }}
      />
    </div>
  );
}

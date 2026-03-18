"use client";

import { ThemeToggle } from "@/components/ui/theme-toggle";
import { cn } from "@/lib/utils";
import { LayoutDashboard, Leaf, LineChart, Menu, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";

const SidebarItem = ({
  icon: Icon,
  label,
  href,
  isActive,
  onClick,
  isCollapsed,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  href: string;
  isActive: boolean;
  onClick?: () => void;
  isCollapsed: boolean;
}) => {
  return (
    <Link href={href} onClick={onClick}>
      <div
        className={cn(
          "flex items-center gap-3 px-3 py-2 rounded-md transition-all relative group overflow-hidden",
          isActive
            ? "bg-primary/10 text-primary font-semibold"
            : "text-muted-foreground hover:text-foreground hover:bg-muted/50",
           isCollapsed && "justify-center px-2"
        )}
        title={isCollapsed ? label : undefined}
      >
        <Icon className={cn("shrink-0", isCollapsed ? "w-5 h-5" : "w-4 h-4")} />
        {!isCollapsed && (
             <span className="text-sm tracking-tight whitespace-nowrap">
                {label}
             </span>
        )}
      </div>
    </Link>
  );
};

export function AppShell({ children }: { children: React.ReactNode }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false); // Mobile
  const [isCollapsed, setIsCollapsed] = useState(false); // Desktop
  const pathname = usePathname();

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);
  const toggleCollapse = () => setIsCollapsed(!isCollapsed);

  const menuItems = [
    { label: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
    { label: "Analyze", href: "/analyze", icon: Leaf },
    { label: "History", href: "/history", icon: LineChart },
  ];

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      {/* Mobile Overlay */}
      <div
        className={cn(
          "fixed inset-0 z-40 bg-black/40 lg:hidden transition-opacity",
          isSidebarOpen ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"
        )}
        onClick={() => setIsSidebarOpen(false)}
      />

      {/* Sidebar - Fix width to 240px per UI-UX.md */}
      <aside
        className={cn(
            "fixed lg:sticky top-0 z-50 h-screen border-r border-border bg-card transition-all duration-200 ease-in-out lg:translate-x-0 overflow-y-auto overflow-x-hidden",
            !isSidebarOpen && "-translate-x-full lg:translate-x-0",
             isCollapsed ? "w-[64px] min-w-[64px]" : "w-[240px] min-w-[240px]"
        )}
      >
        <div className="flex flex-col h-full p-4">
          <div className={cn("flex items-center mb-8", isCollapsed ? "justify-center" : "justify-between")}>
            <Link href="/" className="flex items-center gap-2.5 group">
              <div className="w-8 h-8 rounded-md bg-primary flex items-center justify-center shrink-0">
                <Leaf className="w-5 h-5 text-primary-foreground" />
              </div>
              {!isCollapsed && (
                 <span className="text-xl font-bold tracking-tight whitespace-nowrap">
                    AgroSense
                 </span>
              )}
            </Link>
             {/* Desktop Toggle */}
             {!isCollapsed && (
                <Button variant="ghost" size="icon" className="hidden lg:flex h-8 w-8" onClick={toggleCollapse}>
                    <Menu className="w-4 h-4" />
                </Button>
             )}
             {/* Mobile Toggle */}
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden h-8 w-8"
              onClick={toggleSidebar}
            >
              <X className="w-5 h-5" />
            </Button>
          </div>
          
           {/* Centered Desktop Toggle when collapsed */}
           {isCollapsed && (
                <div className="hidden lg:flex justify-center mb-6">
                     <Button variant="ghost" size="icon" className="h-8 w-8" onClick={toggleCollapse}>
                        <Menu className="w-4 h-4" />
                    </Button>
                </div>
            )}

          <nav className="space-y-1 flex-1">
            {menuItems.map((item) => (
              <SidebarItem
                key={item.href}
                {...item}
                isActive={pathname === item.href}
                onClick={() => setIsSidebarOpen(false)}
                isCollapsed={isCollapsed}
              />
            ))}
          </nav>

          <div className="pt-4 border-t border-border mt-auto">
             <div className={cn("flex items-center justify-between", isCollapsed && "justify-center")}>
                {!isCollapsed && <span className="text-xs text-muted-foreground font-medium">AgroSense v2.0</span>}
                <ThemeToggle />
             </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-background">
        {/* Header - Fix height to h-14 per UI-UX.md */}
        <header className="h-14 border-b bg-background flex items-center justify-between px-6 sticky top-0 z-30">
          <div className="flex items-center gap-3">
             <Button
                variant="ghost"
                size="icon"
                className="lg:hidden h-9 w-9"
                onClick={toggleSidebar}
              >
                <Menu className="w-5 h-5" />
              </Button>
             <div className="hidden lg:flex items-center gap-2 text-sm text-muted-foreground font-medium">
                <span>Platform</span>
                <span className="opacity-30">/</span>
                <span className="text-foreground capitalize">{pathname.split("/")[1] || "Overview"}</span>
             </div>
          </div>

          <div className="flex items-center gap-4">
             <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" title="System Status: Online" />
             <ThemeToggle />
          </div>
        </header>

        {/* Normal Content Padding (24-32px) */}
        <main className="flex-1 p-6 lg:p-10 w-full max-w-[1500px] mx-auto overflow-x-hidden">
           <div className="h-full">
              {children}
           </div>
        </main>
      </div>
    </div>
  );
}

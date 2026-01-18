"use client";

import { motion } from "framer-motion";
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
  icon: React.ElementType;
  label: string;
  href: string;
  isActive: boolean;
  onClick?: () => void;
  isCollapsed: boolean;
}) => {
  return (
    <Link href={href} onClick={onClick}>
      <motion.div
        whileHover={{ x: isCollapsed ? 0 : 4 }}
        className={cn(
          "flex items-center gap-3 px-4 py-3 rounded-lg transition-colors relative group overflow-hidden",
          isActive
            ? "bg-primary/10 text-primary"
            : "text-muted-foreground hover:text-foreground hover:bg-muted/50",
           isCollapsed && "justify-center px-2"
        )}
        title={isCollapsed ? label : undefined}
      >
        {isActive && (
          <motion.div
            layoutId="active-sidebar-item"
            className="absolute left-0 top-0 bottom-0 w-1 bg-primary rounded-r-full"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          />
        )}
        <Icon className="w-5 h-5 shrink-0" />
        {!isCollapsed && (
            <motion.span 
                initial={{ opacity: 0, width: 0 }} 
                animate={{ opacity: 1, width: "auto" }}
                exit={{ opacity: 0, width: 0 }}
                className="font-medium whitespace-nowrap"
            >
                {label}
            </motion.span>
        )}
      </motion.div>
    </Link>
  );
};

export default function AppShell({ children }: { children: React.ReactNode }) {
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
      <motion.div
        initial={false}
        animate={{
          opacity: isSidebarOpen ? 1 : 0,
          pointerEvents: isSidebarOpen ? "auto" : "none",
        }}
        className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm lg:hidden"
        onClick={() => setIsSidebarOpen(false)}
      />

      {/* Sidebar - Fix position logic */}
      <motion.aside
        initial={false}
        animate={{
          width: isCollapsed ? 80 : 280,
        }}
        className={cn(
            "fixed lg:sticky top-0 z-50 h-screen border-r border-border bg-card/50 backdrop-blur-xl transition-transform duration-300 ease-in-out lg:translate-x-0 overflow-y-auto overflow-x-hidden",
            !isSidebarOpen && "-translate-x-full lg:translate-x-0",
             isCollapsed ? "w-[80px] min-w-[80px]" : "w-[280px] min-w-[280px]"
        )}
      >
        <div className="flex flex-col h-full p-6">
          <div className={cn("flex items-center mb-10", isCollapsed ? "justify-center" : "justify-between")}>
            <Link href="/" className="flex items-center gap-2 group">
              <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center shrink-0 group-hover:scale-110 transition-transform">
                <Leaf className="w-5 h-5 text-primary-foreground" />
              </div>
              {!isCollapsed && (
                 <motion.span 
                    initial={{ opacity: 0 }} 
                    animate={{ opacity: 1 }} 
                    className="text-xl font-bold tracking-tight whitespace-nowrap"
                 >
                    AgroSense
                 </motion.span>
              )}
            </Link>
             {/* Desktop Toggle */}
             {!isCollapsed && (
                <Button variant="ghost" size="icon" className="hidden lg:flex" onClick={toggleCollapse}>
                    <Menu className="w-4 h-4" />
                </Button>
             )}
             {/* Mobile Toggle */}
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden"
              onClick={toggleSidebar}
            >
              <X className="w-5 h-5" />
            </Button>
          </div>
          
          {/* Centered Desktop Toggle when collapsed */}
           {isCollapsed && (
                <div className="hidden lg:flex justify-center mb-6">
                     <Button variant="ghost" size="icon" onClick={toggleCollapse}>
                        <Menu className="w-4 h-4" />
                    </Button>
                </div>
            )}

          <nav className="space-y-2 flex-1">
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

          <div className="pt-6 border-t border-border mt-auto">
             <div className={cn("flex items-center", isCollapsed ? "justify-center" : "justify-between px-2")}>
                {!isCollapsed && <span className="text-xs text-muted-foreground mr-2">v2.0.0</span>}
                <ThemeToggle />
             </div>
          </div>
        </div>
      </motion.aside>

      {/* Main Content - Allow natural window scroll */}
      <div className="flex-1 flex flex-col min-h-screen w-full">
        {/* Mobile Header */}
        <header className="lg:hidden flex items-center justify-between p-4 border-b border-border bg-background/80 backdrop-blur-md sticky top-0 z-30">
          <div className="flex items-center gap-2">
             <div className="w-6 h-6 rounded-md bg-primary flex items-center justify-center">
                <Leaf className="w-4 h-4 text-primary-foreground" />
              </div>
              <span className="font-bold">AgroSense</span>
          </div>
          <Button variant="ghost" size="icon" onClick={toggleSidebar}>
            <Menu className="w-5 h-5" />
          </Button>
        </header>

        <main className="flex-1 p-4 lg:p-8 w-full max-w-[1400px] mx-auto">
           <motion.div
                key={pathname}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, ease: "easeOut" }}
                className="h-full"
           >
              {children}
           </motion.div>
        </main>
      </div>
    </div>
  );
}

"use client";

import React, { memo } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

// 模块级常量:稳定引用,避免每次渲染新建数组触发 ReactMarkdown 重复处理。
const remarkPlugins = [remarkGfm];

// 自定义渲染:链接新标签页打开、图片限宽懒加载、表格外包横向滚动容器。
// 解构出 node 避免把它当未知 prop 透传给 DOM 元素。
const components: Components = {
  a: ({ node, ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
  img: ({ node, ...props }) => <img {...props} loading="lazy" />,
  table: ({ node, ...props }) => (
    <div className="md-table-wrap">
      <table {...props} />
    </div>
  ),
};

function MarkdownBase({ content }: { content: string }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown remarkPlugins={remarkPlugins} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
}

// memo:历史/非流式消息 content 不变时跳过重渲染(流式消息 content 每 token 变化仍会重渲染)。
export const Markdown = memo(MarkdownBase);

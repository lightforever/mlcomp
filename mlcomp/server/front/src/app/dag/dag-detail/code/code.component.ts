import {AfterViewInit, Component} from '@angular/core';
import {FlatTreeControl} from '@angular/cdk/tree';
import {MatTreeFlatDataSource, MatTreeFlattener} from '@angular/material/tree';
import {FlatNode, CodeNode} from '../../../models'
import {DagDetailService} from "../dag-detail/dag-detail.service";
import {ActivatedRoute} from "@angular/router";
import {MessageService} from "../../../message.service";
import {DynamicresourceService} from "../../../dynamicresource.service";
import {DomSanitizer} from "@angular/platform-browser";

@Component({
    selector: 'app-code',
    templateUrl: './code.component.html',
    styleUrls: ['./code.component.css']
})

export class CodeComponent implements AfterViewInit {

    public dag: number;
    private current_node: FlatNode;
    private edit_mode: boolean;

    private transformer = (node: CodeNode, level: number) => {
        return {
            expandable: !!node.children && node.children.length > 0,
            name: node.name,
            level: level,
            content: node.content,
            id: node.id,
            dag: node.dag,
            storage: node.storage
        };
    };

    treeControl = new FlatTreeControl<FlatNode>(
        node => node.level, node => node.expandable);

    treeFlattener = new MatTreeFlattener(
        this.transformer,
        node => node.level,
        node => node.expandable,
        node => node.children);

    dataSource = new MatTreeFlatDataSource(this.treeControl,
        this.treeFlattener);

    constructor(private service: DagDetailService,
                private route: ActivatedRoute,
                private message_service: MessageService,
                private resource_service: DynamicresourceService,
                private sanitizer: DomSanitizer
    ) {

    }

    ngAfterViewInit() {
        let self = this;
        this.service.get_code(this.dag).subscribe(res => {
            self.dataSource.data = res.items;
        });
        this.resource_service.load('prettify',
            'prettify-yaml',
            'prettify-css')
    }

    prettify_lang(ext: string) {
        switch (ext) {
            case 'py':
                return 'lang-py';
            case 'yaml':
            case 'yml':
                return 'lang-yaml';
            case 'json':
                return 'lang-json';
            default:
                return ''
        }
    }

    node_click(node: FlatNode) {
        let pre = document.createElement('pre');
        pre.textContent = node.content;
        let ext = node.name.indexOf('.') != -1 ?
            node.name.split('.')[1].toLowerCase() : '';
        pre.className = "prettyprint linenums " + this.prettify_lang(ext);
        let code_holder = document.getElementById('codeholder');
        code_holder.innerHTML = '';
        code_holder.appendChild(pre);

        window['PR'].prettyPrint();
        this.current_node = node;
    }

    hasChild = (_: number, node: FlatNode) => node.expandable;


    download() {
        this.service.code_download(this.dag).subscribe(x => {
            let url = window.URL.createObjectURL(x);
            let link = document.createElement('a');
            link.setAttribute('download', String(this.dag));
            link.setAttribute('href', url);
            document.body.append(link);

            link.click();

            document.body.removeChild(link);
        });
    }

    code_edit_click() {
        if (!this.current_node) {
            return;
        }

        let node = this.current_node;
        let code_holder = document.getElementById('codeholder');
        let pre = document.createElement('textarea');
        let height = code_holder.clientHeight;

        pre.setAttribute('style',
            `width:100%; height:${height}px;display:block`);

        pre.textContent = node.content;
        code_holder.innerHTML = '';
        code_holder.appendChild(pre);
        this.edit_mode = true;
    }

    code_td_click(event) {
        if (!this.current_node) {
            return;
        }
        if (event.target.type == 'textarea') {
            return;
        }
        let code_holder = document.getElementById('codeholder');
        if (code_holder && code_holder.children.length > 0) {
            if (code_holder.children[0].tagName == 'TEXTAREA') {
                // @ts-ignore
                this.current_node.content = code_holder.children[0].value;
                this.service.update_code(
                    this.current_node.id,
                    this.current_node.content,
                    this.current_node.dag,
                    this.current_node.storage
                ).subscribe(x => {
                    this.current_node.id = x.file
                });
                this.node_click(this.current_node);
                this.edit_mode = false;
                return;
            }
        }
    }

    has_parent_id(element, id) {
        return element.id == id ||
            (element.parentNode && this.has_parent_id(element.parentNode, id))
    }

    code_click(event) {
        if (this.has_parent_id(event.target, 'codeholder')) {
            if (!this.edit_mode && !window.getSelection().toString()) {
                this.code_edit_click();
            }
        } else {
            this.code_td_click(event);
        }
    }
}

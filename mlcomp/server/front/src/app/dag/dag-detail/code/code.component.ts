import {AfterViewInit, Component, OnInit} from '@angular/core';
import {FlatTreeControl} from '@angular/cdk/tree';
import {MatTreeFlatDataSource, MatTreeFlattener} from '@angular/material/tree';
import {FlatNode, CodeNode} from '../../../models'
import {DagDetailService} from "../dag-detail/dag-detail.service";
import {ActivatedRoute} from "@angular/router";
import {MessageService} from "../../../message.service";
import {DynamicresourceService} from "../../../dynamicresource.service";

@Component({
    selector: 'app-code',
    templateUrl: './code.component.html',
    styleUrls: ['./code.component.css']
})

export class CodeComponent implements AfterViewInit {

    public dag: number;

    private transformer = (node: CodeNode, level: number) => {
        return {
            expandable: !!node.children && node.children.length > 0,
            name: node.name,
            level: level,
            content: node.content,
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
                private resource_service: DynamicresourceService
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
        let ext = node.name.indexOf('.')!=-1?
            node.name.split('.')[1].toLowerCase():'';
        pre.className = "prettyprint linenums " + this.prettify_lang(ext);
        let code_holder = document.getElementById('codeholder');
        code_holder.innerHTML = '';
        code_holder.appendChild(pre);

        window['PR'].prettyPrint();
    }

    hasChild = (_: number, node: FlatNode) => node.expandable;


}

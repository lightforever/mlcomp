import {Component, OnInit} from '@angular/core';
import {FlatTreeControl} from '@angular/cdk/tree';
import {MatTreeFlatDataSource, MatTreeFlattener} from '@angular/material/tree';
import {FlatNode, CodeNode} from '../../../models'
import {DagDetailService} from "../../../dag-detail.service";
import {ActivatedRoute} from "@angular/router";
import {MessageService} from "../../../message.service";

@Component({
    selector: 'app-code',
    templateUrl: './code.component.html',
    styleUrls: ['./code.component.css']
})

export class CodeComponent implements OnInit {

    private dag_id: string;

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
        this.transformer, node => node.level, node => node.expandable, node => node.children);

    dataSource = new MatTreeFlatDataSource(this.treeControl, this.treeFlattener);

    constructor(private service: DagDetailService, private route: ActivatedRoute,
                private message_service: MessageService
    ) {

    }

    ngOnInit() {
        let self = this;
        this.dag_id = this.route.parent.snapshot.paramMap.get('id');
        this.service.get_code(this.dag_id).subscribe(res => {
            self.dataSource.data = res;
        });

        this.load_prettify();
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
        let ext = node.name.indexOf('.')!=-1?node.name.split('.')[1].toLowerCase():'';
        pre.className = "prettyprint linenums " + this.prettify_lang(ext);
        let code_holder = document.getElementById('codeholder');
        code_holder.innerHTML = '';
        code_holder.appendChild(pre);

        window['PR'].prettyPrint();
    }

    load_prettify() {
        let self = this;
        this.message_service.add('loading prettify');
        let scripts_to_load = ['assets/prettify/prettify.js', 'assets/prettify/lang-yaml.js'];
        for (let s of scripts_to_load) {
            let node = document.createElement('script');
            node.src = s;
            node.type = 'text/javascript';
            node.async = true;
            document.getElementsByTagName('head')[0].appendChild(node);
        }


        let node2 = document.createElement('link');
        node2.href = "assets/prettify/prettify.css";
        node2.type = 'text/css';
        node2.rel = 'stylesheet';
        document.getElementsByTagName('head')[0].appendChild(node2);
    }


    hasChild = (_: number, node: FlatNode) => node.expandable;


}

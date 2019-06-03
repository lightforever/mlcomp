import {Component, OnInit, ViewChild, EventEmitter, Input, ElementRef, OnDestroy} from '@angular/core';
import {MatSort, MatTableDataSource, MatPaginator} from '@angular/material';
import {Observable, of as observableOf, merge} from 'rxjs';
import {catchError} from 'rxjs/operators';
import {map} from 'rxjs/operators';
import {startWith} from 'rxjs/operators';
import {switchMap} from 'rxjs/operators';
import {Location} from '@angular/common';
import {PaginatorFilter, PaginatorRes} from "./models";
import {BaseService} from "./base.service";

export abstract class Paginator<T> implements OnInit, OnDestroy {
    dataSource: MatTableDataSource<T> = new MatTableDataSource();

    @ViewChild(MatPaginator) paginator: MatPaginator;
    @ViewChild(MatSort) sort: MatSort;
    change: EventEmitter<any> = new EventEmitter();
    data_updated: EventEmitter<any> = new EventEmitter();

    protected abstract displayed_columns: string[];
    protected default_page_size: number = 15;
    isLoading_results = false;
    total: number;
    private interval: number;
    id_column: string = 'id';

    constructor(
        protected service: BaseService,
        protected location: Location,
        protected enable_interval: boolean = true
    ) {

    }

    protected _ngOnInit() {

    }

    get_filter(): any {
        let res = new PaginatorFilter();
        res.page_number = this.paginator ? this.paginator.pageIndex : 0;
        res.page_size = this.paginator ? this.paginator.pageSize || this.default_page_size : 10;
        if (this.sort) {
            res.sort_column = this.sort.active ? this.sort.active : '';
            res.sort_descending = this.sort.direction ? this.sort.direction == 'desc' : true;
        }

        return res;
    }

    normalizeArray<T>(array: Array<T>) {
        const normalizedObject: any = {};
        for (let i = 0; i < array.length; i++) {
            const key = 'id' + array[i][this.id_column].toString();
            normalizedObject[key] = array[i]
        }
        return normalizedObject as { [key: string]: T }
    }

    sync_objects(source, target) {
        for (let name in source) {
            if (JSON.stringify(source[name]) != JSON.stringify(target[name])) {
                target[name] = source[name];
            }
        }
    }

    ngOnInit() {
        this._ngOnInit();

        // If the user changes the sort order, reset back to the first page.
        if(this.sort){
            this.sort.sortChange.subscribe(() => this.paginator.pageIndex = 0);
        }


        let m = merge(this.paginator.page, this.change);
        if(this.sort){
            m = merge(m, this.sort.sortChange);
        }
        m.pipe(
            startWith({}),
            switchMap(() => {
                this.isLoading_results = true;
                let filter = this.get_filter();
                if(!filter){
                    return observableOf(new PaginatorRes<T>());
                }
                return this.service.get_paginator<T>(filter);
            }),
            map(res => {
                // Flip flag to show that loading has finished.
                this.isLoading_results = false;

                return res;
            }),
            catchError(() => {
                this.isLoading_results = false;
                return observableOf(new PaginatorRes<T>());
            })
        ).subscribe(res => {
            if (!res.data) {
                return;
            }
            if (this.dataSource.data && res.data.length == this.dataSource.data.length) {
                let res_d = this.normalizeArray(res.data);
                let target_d = this.normalizeArray(this.dataSource.data);
                let names = Object.getOwnPropertyNames(res_d);
                for (let k of names) {
                    if (k in target_d) {
                        this.sync_objects(res_d[k], target_d[k]);
                        delete res_d[k];
                        delete target_d[k];
                    }

                }

                let data = this.dataSource.data.slice(0);
                for (let k in target_d) {
                    let index = data.indexOf(target_d[k]);
                    data.splice(index, 1);
                }
                let res_a = [];
                for (let k in res_d) {
                    res_a.push(res_d[k]);
                }
                for (let i = 0; i < res_a.length; i++) {
                    data.splice(i, 0, res_a[i]);
                }

                this.dataSource.data = data;

            } else {
                this.dataSource.data = res.data;
                this.total = res.total;
            }

            this.data_updated.emit(res);
        });

        if (this.enable_interval) {
            this.interval = setInterval(() => this.change.emit('event'), 3000);
        }

    }

    ngOnDestroy() {
        if (this.enable_interval) {
            clearInterval(this.interval);
        }

    }

}
